#!/usr/bin/env python3
"""
Final Entity Resolution Pipeline
=================================
Complete end-to-end pipeline:
  1. Preprocess test data (if needed)
  2. Enhanced retrieval (V3) with additional channels
  3. Memory-safe streaming feature extraction via DuckDB
  4. Hard negative LightGBM training
  5. Threshold-optimized 0/1/MANY decision
  6. Final test inference and output generation
"""

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import duckdb
import json
import time
import gc
import os
import re
import sys
import joblib
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz, process as rfprocess
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode
import jellyfish
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution"
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = CODE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DIR = ARTIFACTS_DIR / "processed_v2"
SPLITS_DIR = ARTIFACTS_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 0: Check if test data is preprocessed
# ============================================================
def preprocess_source(input_path, output_path):
    """Preprocess a raw TSV source file to normalized parquet."""
    print(f"  Preprocessing {input_path.name}...")
    sys.path.insert(0, str(CODE_DIR))
    from src.preprocessing.normalize_v2 import process_batch

    # Read in chunks to limit memory
    chunks = pd.read_csv(input_path, sep='\t', dtype=str, chunksize=500_000)
    all_parts = []
    for i, chunk in enumerate(chunks):
        print(f"    chunk {i}...")
        chunk = process_batch(chunk)
        if 'name_tokens' in chunk.columns:
            chunk['name_tokens'] = chunk['name_tokens'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else '')
        if 'address_tokens' in chunk.columns:
            chunk['address_tokens'] = chunk['address_tokens'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else '')
        all_parts.append(pl.from_pandas(chunk))
        del chunk
        gc.collect()

    pl.concat(all_parts).write_parquet(output_path)
    print(f"  Saved {output_path.name}")
    del all_parts
    gc.collect()


def ensure_preprocessed():
    """Ensure all needed parquet files exist."""
    needed = {
        "train_s1_split": (SPLITS_DIR / "train_s1_split.tsv", PROCESSED_DIR / "train_s1_split.parquet"),
        "test_s1": (DATASET_DIR / "test" / "test_source1.tsv", PROCESSED_DIR / "test_s1.parquet"),
        "test_s2": (DATASET_DIR / "test" / "test_source2.tsv", PROCESSED_DIR / "test_s2.parquet"),
        "test_s3": (DATASET_DIR / "test" / "test_source3.tsv", PROCESSED_DIR / "test_s3.parquet"),
    }

    for name, (src, dst) in needed.items():
        if not dst.exists():
            print(f"Preprocessing {name}...")
            preprocess_source(src, dst)
        else:
            print(f"  {name} already preprocessed.")

    # Ensure train S1 split parquet exists
    train_s1_pq = PROCESSED_DIR / "train_s1_split.parquet"
    if not train_s1_pq.exists():
        preprocess_source(SPLITS_DIR / "train_s1_split.tsv", train_s1_pq)


# ============================================================
# STEP 1: ENHANCED RETRIEVAL V3
# ============================================================
def retrieval_v3(s1_path, s2_path, s3_path, output_path, gt_path=None, metrics_path=None):
    """
    Enhanced multi-channel retrieval with:
    - Exact normalized name (cap 2000)
    - Exact normalized name without legal suffixes (cap 2000)
    - Name token blocking (cap 1000)
    - Name bigram blocking (uncapped - naturally sparse)
    - Address number + name token intersection (uncapped)
    - Postal code + name token intersection (uncapped)
    - Phonetic (Soundex) name blocking
    - Sorted token name blocking
    """
    print("=" * 60)
    print("RETRIEVAL V3: Enhanced Multi-Channel")
    print("=" * 60)
    t0 = time.time()

    cols = ["entity_id", "country_norm", "name_legal", "name_punct",
            "name_sorted", "address_norm", "address_numbers", "name_tokens"]

    s1 = pl.read_parquet(s1_path, columns=cols)
    s2 = pl.read_parquet(s2_path, columns=cols)
    s3 = pl.read_parquet(s3_path, columns=cols)

    print(f"  S1: {s1.height}, S2: {s2.height}, S3: {s3.height}")

    candidates = []

    def do_channel(name, s1_df, s2_df, s3_df, keys, channel_name, max_freq=None):
        """Run a blocking channel with optional frequency cap."""
        s1_df = s1_df.select(keys + ["entity_id"])
        s2_df = s2_df.select(keys + ["entity_id"])
        s3_df = s3_df.select(keys + ["entity_id"])
        if max_freq is not None:
            # Calculate document frequency across S2+S3
            freq2 = s2_df.group_by(keys).len()
            freq3 = s3_df.group_by(keys).len()
            freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
            valid = freq.filter((pl.col("len") > 0) & (pl.col("len") <= max_freq)).select(keys)
            s1_f = s1_df.join(valid, on=keys, how="inner")
            s2_f = s2_df.join(valid, on=keys, how="inner")
            s3_f = s3_df.join(valid, on=keys, how="inner")
        else:
            s1_f, s2_f, s3_f = s1_df, s2_df, s3_df

        c_s2 = s1_f.join(s2_f, on=keys, how="inner").select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_right").alias("candidate_entity_id")
        ]).with_columns(pl.lit(channel_name).alias("channel"))

        c_s3 = s1_f.join(s3_f, on=keys, how="inner").select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_right").alias("candidate_entity_id")
        ]).with_columns(pl.lit(channel_name).alias("channel"))

        total = c_s2.height + c_s3.height
        print(f"  {name}: {total:,} pairs")
        candidates.extend([c_s2, c_s3])
        return total

    # CH 1: Exact normalized name (with legal suffixes stripped) + country
    print("CH 1: Exact Legal Name")
    s1_c = s1.filter(pl.col("name_legal") != "")
    s2_c = s2.filter(pl.col("name_legal") != "")
    s3_c = s3.filter(pl.col("name_legal") != "")
    do_channel("exact_legal", s1_c, s2_c, s3_c,
               ["name_legal", "country_norm"], "exact_name", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    # CH 2: Exact name_punct (with legal suffixes) + country
    print("CH 2: Exact Punct Name")
    s1_c = s1.filter(pl.col("name_punct") != "")
    s2_c = s2.filter(pl.col("name_punct") != "")
    s3_c = s3.filter(pl.col("name_punct") != "")
    do_channel("exact_punct", s1_c, s2_c, s3_c,
               ["name_punct", "country_norm"], "exact_punct", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    # CH 3: Exact Address + country
    print("CH 3: Exact Address")
    s1_c = s1.filter(pl.col("address_norm") != "")
    s2_c = s2.filter(pl.col("address_norm") != "")
    s3_c = s3.filter(pl.col("address_norm") != "")
    do_channel("exact_addr", s1_c, s2_c, s3_c,
               ["address_norm", "country_norm"], "exact_addr", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    # CH 4: Sorted-token name + country (catches reordered names)
    print("CH 4: Sorted Name")
    s1_c = s1.filter(pl.col("name_sorted") != "")
    s2_c = s2.filter(pl.col("name_sorted") != "")
    s3_c = s3.filter(pl.col("name_sorted") != "")
    do_channel("sorted_name", s1_c, s2_c, s3_c,
               ["name_sorted", "country_norm"], "sorted_name", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    # Prepare token exploded data
    def prepare_tokens(df):
        return (df.filter(pl.col("name_tokens") != "")
                .with_columns(pl.col("name_tokens").str.split(" "))
                .explode("name_tokens")
                .rename({"name_tokens": "token"})
                .filter(pl.col("token") != "")
                .filter(pl.col("token").str.len_chars() > 1))  # skip single-char tokens

    s1_tok = prepare_tokens(s1)
    s2_tok = prepare_tokens(s2)
    s3_tok = prepare_tokens(s3)

    # CH 5: Name Token blocking (cap 1000)
    print("CH 5: Token Blocking (cap 1000)")
    keys = ["token", "country_norm"]
    freq2 = s2_tok.group_by(keys).len()
    freq3 = s3_tok.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_tokens = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_tf = s1_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")
    s2_tf = s2_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")
    s3_tf = s3_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")

    c_s2 = s1_tf.join(s2_tf, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("token").alias("channel"))
    c_s3 = s1_tf.join(s3_tf, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("token").alias("channel"))
    print(f"  token: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_tf, s2_tf, s3_tf, c_s2, c_s3, valid_tokens, freq, freq2, freq3; gc.collect()

    # CH 6: Name Token + Address Number intersection (capped)
    print("CH 6: Token + Address Number")
    s1_ta = s1_tok.filter(pl.col("address_numbers") != "")
    s2_ta = s2_tok.filter(pl.col("address_numbers") != "")
    s3_ta = s3_tok.filter(pl.col("address_numbers") != "")
    keys = ["token", "address_numbers", "country_norm"]
    
    freq2 = s2_ta.group_by(keys).len()
    freq3 = s3_ta.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_ta = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 1000)).select(keys)

    c_s2 = s1_ta.select(keys + ["entity_id"]).join(valid_ta, on=keys, how="inner").join(s2_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    c_s3 = s1_ta.select(keys + ["entity_id"]).join(valid_ta, on=keys, how="inner").join(s3_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    print(f"  tok_addrnum: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_ta, s2_ta, s3_ta, c_s2, c_s3, valid_ta, freq, freq2, freq3; gc.collect()

    # CH 7: Postal Code + Name Token (capped)
    print("CH 7: Postal + Token")
    s1_postal = s1_tok.with_columns(
        pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s2_postal = s2_tok.with_columns(
        pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s3_postal = s3_tok.with_columns(
        pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    keys = ["token", "postal", "country_norm"]
    
    freq2 = s2_postal.group_by(keys).len()
    freq3 = s3_postal.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_postal = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 1000)).select(keys)

    c_s2 = s1_postal.select(keys + ["entity_id"]).join(valid_postal, on=keys, how="inner").join(s2_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    c_s3 = s1_postal.select(keys + ["entity_id"]).join(valid_postal, on=keys, how="inner").join(s3_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    print(f"  tok_postal: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_postal, s2_postal, s3_postal, c_s2, c_s3, valid_postal, freq, freq2, freq3; gc.collect()

    # CH 8: Name bigram blocking (pairs of tokens, uncapped - naturally sparse)
    print("CH 8: Token Bigram (uncapped)")
    def make_bigrams(tok_df):
        """Generate bigrams from token-exploded DF."""
        # Group tokens by entity, then self-join to get pairs
        ent_toks = tok_df.select(["entity_id", "token", "country_norm"]).unique()
        # Get pairs via self-join on entity_id
        pairs = ent_toks.join(ent_toks, on=["entity_id", "country_norm"], how="inner", suffix="_2")
        # Keep only ordered pairs (token < token_2) to avoid duplicates
        pairs = pairs.filter(pl.col("token") < pl.col("token_2"))
        # Create bigram key
        pairs = pairs.with_columns(
            (pl.col("token") + "|" + pl.col("token_2")).alias("bigram")
        ).select(["entity_id", "country_norm", "bigram"])
        return pairs

    # Only use rare bigrams (cap 500)
    s1_bg = make_bigrams(s1_tok)
    s2_bg = make_bigrams(s2_tok)
    s3_bg = make_bigrams(s3_tok)

    keys = ["bigram", "country_norm"]
    freq2 = s2_bg.group_by(keys).len()
    freq3 = s3_bg.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_bg = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_bg = s1_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")
    s2_bg = s2_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")
    s3_bg = s3_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")

    c_s2 = s1_bg.join(s2_bg, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("bigram").alias("channel"))
    c_s3 = s1_bg.join(s3_bg, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("bigram").alias("channel"))
    print(f"  bigram: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_bg, s2_bg, s3_bg, c_s2, c_s3, valid_bg, freq, freq2, freq3; gc.collect()

    del s1_tok, s2_tok, s3_tok; gc.collect()

    # CH 9: Phonetic (Soundex of first significant token)
    print("CH 9: Phonetic (Soundex)")
    def add_soundex(df):
        """Add soundex of first token of name_legal."""
        names = df["name_legal"].to_list()
        soundex_codes = []
        for n in names:
            if n and n.strip():
                first_tok = n.split()[0] if n.split() else ""
                if first_tok and first_tok.isalpha():
                    try:
                        soundex_codes.append(jellyfish.soundex(first_tok))
                    except:
                        soundex_codes.append("")
                else:
                    soundex_codes.append("")
            else:
                soundex_codes.append("")
        return df.with_columns(pl.Series("soundex", soundex_codes))

    s1_sx = add_soundex(s1).filter(pl.col("soundex") != "")
    s2_sx = add_soundex(s2).filter(pl.col("soundex") != "")
    s3_sx = add_soundex(s3).filter(pl.col("soundex") != "")

    keys = ["soundex", "country_norm"]
    # Cap at 400 to avoid generic soundex codes
    freq2 = s2_sx.group_by(keys).len()
    freq3 = s3_sx.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_sx = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_sx = s1_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")
    s2_sx = s2_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")
    s3_sx = s3_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")

    c_s2 = s1_sx.join(s2_sx, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("phonetic").alias("channel"))
    c_s3 = s1_sx.join(s3_sx, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("phonetic").alias("channel"))
    print(f"  phonetic: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_sx, s2_sx, s3_sx, c_s2, c_s3; gc.collect()

    # UNION & DEDUPLICATE
    print("Union and deduplicate...")
    # Write to a temporary parquet to save memory, then read it back
    pl.concat(candidates).write_parquet(str(output_path) + "_tmp.parquet")
    del candidates
    gc.collect()

    all_cands = pl.scan_parquet(str(output_path) + "_tmp.parquet").group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        pl.col("channel").unique()
    ).with_columns(
        pl.col("channel").list.join("|").alias("channels"),
        pl.col("channel").list.len().cast(pl.UInt8).alias("num_channels")
    ).drop("channel").collect(engine="streaming")
    
    import os
    try:
        os.remove(str(output_path) + "_tmp.parquet")
    except:
        pass

    t_ret = time.time() - t0
    print(f"Retrieval V3 finished in {t_ret:.1f}s. Total candidate pairs: {all_cands.height:,}")

    # Candidate statistics
    cand_per_s1 = all_cands.group_by("source1_entity_id").len()
    stats = cand_per_s1["len"].describe()
    print(f"  Candidates/S1 stats:")
    print(f"    Mean: {cand_per_s1['len'].mean():.1f}")
    print(f"    Median: {cand_per_s1['len'].median():.1f}")
    p95 = cand_per_s1["len"].quantile(0.95)
    p99 = cand_per_s1["len"].quantile(0.99)
    print(f"    P95: {p95:.0f}, P99: {p99:.0f}, Max: {cand_per_s1['len'].max()}")

    all_cands.write_parquet(output_path)
    print(f"Saved candidates to {output_path}")

    # Evaluate if GT available
    if gt_path and metrics_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")

        total_true = gt_exploded.height

        recovered = gt_exploded.join(
            all_cands.select(["source1_entity_id", "candidate_entity_id"]),
            left_on=["source1_entity_id", "matched_entity_ids"],
            right_on=["source1_entity_id", "candidate_entity_id"],
            how="inner"
        )
        recovered_links = recovered.height
        pair_recall = recovered_links / total_true if total_true > 0 else 0

        # Entity recall: how many S1 entities have at least one true match recovered
        s1_with_matches = gt.filter(pl.col("matched_entity_ids") != "")
        s1_with_any_recovered = recovered.select("source1_entity_id").unique()
        entity_recall = s1_with_any_recovered.height / s1_with_matches.height if s1_with_matches.height > 0 else 0

        metrics = {
            "total_candidates": all_cands.height,
            "total_true_links": total_true,
            "recovered_links": recovered_links,
            "pair_recall": pair_recall,
            "entity_recall": entity_recall,
            "runtime": t_ret,
            "mean_candidates_per_s1": float(cand_per_s1['len'].mean()),
            "p95_candidates": float(p95),
            "p99_candidates": float(p99),
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  Pair Recall: {pair_recall:.4f} ({recovered_links}/{total_true})")
        print(f"  Entity Recall (at-least-one): {entity_recall:.4f}")

    del all_cands; gc.collect()
    return t_ret


# ============================================================
# STEP 2: MEMORY-SAFE FEATURE EXTRACTION via DuckDB
# ============================================================
def compute_features_duckdb(candidates_path, s1_path, s2_path, s3_path,
                            output_path, gt_path=None, chunk_size=100_000):
    """
    Streaming feature extraction using DuckDB for joins and
    RapidFuzz batch API for string similarity.
    """
    print("=" * 60)
    print("FEATURE EXTRACTION (DuckDB + RapidFuzz batch)")
    print("=" * 60)
    t0 = time.time()

    con = duckdb.connect()

    # Register parquet files as views
    con.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet('{candidates_path}')")
    con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{s1_path}')")

    # Union S2 and S3 into a single source view
    con.execute(f"""
        CREATE VIEW sources AS
        SELECT * FROM read_parquet('{s2_path}')
        UNION ALL
        SELECT * FROM read_parquet('{s3_path}')
    """)

    total_candidates = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"  Total candidates: {total_candidates:,}")

    # Prepare GT labels
    gt_set = set()
    if gt_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        for row in gt_exploded.iter_rows():
            gt_set.add((row[0], row[1]))
        print(f"  GT links loaded: {len(gt_set)}")

    all_chunk_files = []
    chunk_idx = 0

    for offset in range(0, total_candidates, chunk_size):
        t_chunk = time.time()

        # Fetch chunk of candidates with joined features via DuckDB
        query = f"""
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.channels,
            c.num_channels,
            COALESCE(s1.name_punct, '') as name1,
            COALESCE(s1.name_legal, '') as name1_legal,
            COALESCE(s1.name_sorted, '') as name1_sorted,
            COALESCE(s1.address_norm, '') as addr1,
            COALESCE(s1.address_numbers, '') as addr_nums1,
            COALESCE(s1.country_norm, '') as country1,
            COALESCE(s1.name_tokens, '') as tokens1,
            COALESCE(src.name_punct, '') as name2,
            COALESCE(src.name_legal, '') as name2_legal,
            COALESCE(src.name_sorted, '') as name2_sorted,
            COALESCE(src.address_norm, '') as addr2,
            COALESCE(src.address_numbers, '') as addr_nums2,
            COALESCE(src.country_norm, '') as country2,
            COALESCE(src.name_tokens, '') as tokens2
        FROM (SELECT * FROM candidates LIMIT {chunk_size} OFFSET {offset}) c
        LEFT JOIN s1 ON c.source1_entity_id = s1.entity_id
        LEFT JOIN sources src ON c.candidate_entity_id = src.entity_id
        """

        df = con.execute(query).pl()

        if df.height == 0:
            break

        # STAGE A: CHEAP FEATURES (all vectorized in Polars)
        df = df.with_columns([
            # Exact matches
            (pl.col("name1") == pl.col("name2")).cast(pl.Float32).alias("exact_name_match"),
            (pl.col("name1_legal") == pl.col("name2_legal")).cast(pl.Float32).alias("exact_legal_match"),
            (pl.col("name1_sorted") == pl.col("name2_sorted")).cast(pl.Float32).alias("sorted_name_match"),
            (pl.col("addr1") == pl.col("addr2")).cast(pl.Float32).alias("exact_address_match"),
            (pl.col("country1") == pl.col("country2")).cast(pl.Float32).alias("country_match"),
            (pl.col("addr_nums1") == pl.col("addr_nums2")).cast(pl.Float32).alias("addr_nums_match"),

            # Length features
            pl.col("name1").str.len_chars().cast(pl.Float32).alias("name1_len"),
            pl.col("name2").str.len_chars().cast(pl.Float32).alias("name2_len"),
            pl.col("addr1").str.len_chars().cast(pl.Float32).alias("addr1_len"),
            pl.col("addr2").str.len_chars().cast(pl.Float32).alias("addr2_len"),

            # Retrieval channel features
            pl.col("channels").str.contains("exact_name").cast(pl.Float32).alias("retrieved_by_name"),
            pl.col("channels").str.contains("exact_addr").cast(pl.Float32).alias("retrieved_by_addr"),
            pl.col("channels").str.contains("exact_punct").cast(pl.Float32).alias("retrieved_by_punct"),
            pl.col("channels").str.contains("token").cast(pl.Float32).alias("retrieved_by_token"),
            pl.col("channels").str.contains("bigram").cast(pl.Float32).alias("retrieved_by_bigram"),
            pl.col("channels").str.contains("phonetic").cast(pl.Float32).alias("retrieved_by_phonetic"),
            pl.col("channels").str.contains("sorted").cast(pl.Float32).alias("retrieved_by_sorted"),
            pl.col("channels").str.contains("postal").cast(pl.Float32).alias("retrieved_by_postal"),
            pl.col("num_channels").cast(pl.Float32),
        ])

        # Length ratio/difference
        df = df.with_columns([
            (pl.col("name1_len") - pl.col("name2_len")).abs().alias("name_len_diff"),
            (pl.when(pl.col("name2_len") > 0)
             .then(pl.col("name1_len") / pl.col("name2_len"))
             .otherwise(0.0)).cast(pl.Float32).alias("name_len_ratio"),
            (pl.col("addr1_len") - pl.col("addr2_len")).abs().alias("addr_len_diff"),
        ])

        # Token-level features via Polars list operations
        df = df.with_columns([
            pl.col("tokens1").str.split(" ").alias("t1_list"),
            pl.col("tokens2").str.split(" ").alias("t2_list"),
            pl.col("addr1").str.split(" ").alias("a1_list"),
            pl.col("addr2").str.split(" ").alias("a2_list"),
        ])

        df = df.with_columns([
            # Name token Jaccard
            (pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32) /
             pl.col("t1_list").list.set_union(pl.col("t2_list")).list.len().cast(pl.Float32).replace(0, 1)
             ).fill_nan(0.0).alias("name_jaccard"),
            # Name token containment (fraction of S1 tokens in S2)
            (pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32) /
             pl.col("t1_list").list.len().cast(pl.Float32).replace(0, 1)
             ).fill_nan(0.0).alias("name_containment"),
            # Name shared token count
            pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32).alias("name_shared_tokens"),
            # Address token Jaccard
            (pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32) /
             pl.col("a1_list").list.set_union(pl.col("a2_list")).list.len().cast(pl.Float32).replace(0, 1)
             ).fill_nan(0.0).alias("addr_jaccard"),
            # Address token containment
            (pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32) /
             pl.col("a1_list").list.len().cast(pl.Float32).replace(0, 1)
             ).fill_nan(0.0).alias("addr_containment"),
            # Address shared tokens
            pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32).alias("addr_shared_tokens"),
        ])

        # STAGE B: EXPENSIVE STRING FEATURES (RapidFuzz batch)
        name1_list = df["name1"].to_list()
        name2_list = df["name2"].to_list()
        addr1_list = df["addr1"].to_list()
        addr2_list = df["addr2"].to_list()

        n = len(name1_list)

        # Use RapidFuzz batch/vectorized APIs
        name_ratios = np.zeros(n, dtype=np.float32)
        name_partial = np.zeros(n, dtype=np.float32)
        name_jw = np.zeros(n, dtype=np.float32)
        addr_ratios = np.zeros(n, dtype=np.float32)

        # Process in sub-batches of 50k for memory efficiency
        sub_batch = 50_000
        for i in range(0, n, sub_batch):
            end = min(i + sub_batch, n)
            for j in range(i, end):
                n1, n2 = name1_list[j], name2_list[j]
                a1, a2 = addr1_list[j], addr2_list[j]
                if n1 and n2:
                    name_ratios[j] = fuzz.ratio(n1, n2) / 100.0
                    name_partial[j] = fuzz.partial_ratio(n1, n2) / 100.0
                    name_jw[j] = JaroWinkler.similarity(n1, n2)
                if a1 and a2:
                    addr_ratios[j] = fuzz.ratio(a1, a2) / 100.0

        df = df.with_columns([
            pl.Series("name_fuzz_ratio", name_ratios),
            pl.Series("name_partial_ratio", name_partial),
            pl.Series("name_jaro_winkler", name_jw),
            pl.Series("addr_fuzz_ratio", addr_ratios),
        ])

        # Label assignment
        if gt_set:
            s1_ids = df["source1_entity_id"].to_list()
            c_ids = df["candidate_entity_id"].to_list()
            labels = np.array([1 if (s1_ids[i], c_ids[i]) in gt_set else 0
                               for i in range(len(s1_ids))], dtype=np.int32)
            df = df.with_columns(pl.Series("label", labels))
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("label"))

        # Select feature columns
        feature_cols = [
            "source1_entity_id", "candidate_entity_id", "label",
            "exact_name_match", "exact_legal_match", "sorted_name_match",
            "exact_address_match", "country_match", "addr_nums_match",
            "name_len_diff", "name_len_ratio", "addr_len_diff",
            "retrieved_by_name", "retrieved_by_addr", "retrieved_by_punct",
            "retrieved_by_token", "retrieved_by_bigram", "retrieved_by_phonetic",
            "retrieved_by_sorted", "retrieved_by_postal", "num_channels",
            "name_jaccard", "name_containment", "name_shared_tokens",
            "addr_jaccard", "addr_containment", "addr_shared_tokens",
            "name_fuzz_ratio", "name_partial_ratio", "name_jaro_winkler",
            "addr_fuzz_ratio",
        ]

        chunk_df = df.select(feature_cols)
        chunk_file = str(output_path).replace(".parquet", f"_chunk_{chunk_idx}.parquet")
        chunk_df.write_parquet(chunk_file)
        all_chunk_files.append(chunk_file)

        elapsed = time.time() - t_chunk
        rate = df.height / elapsed if elapsed > 0 else 0
        pos_count = int(chunk_df["label"].sum()) if gt_set else 0
        print(f"  Chunk {chunk_idx} ({offset:,}-{offset+df.height:,}): "
              f"{df.height:,} rows, {pos_count} pos, {rate:.0f} rows/s, {elapsed:.1f}s")

        chunk_idx += 1
        del df, chunk_df, name1_list, name2_list, addr1_list, addr2_list
        del name_ratios, name_partial, name_jw, addr_ratios
        gc.collect()

    # Concatenate all chunks
    print("Concatenating feature chunks...")
    result = pl.concat([pl.read_parquet(f) for f in all_chunk_files])
    result.write_parquet(output_path)

    # Cleanup
    for f in all_chunk_files:
        try:
            os.remove(f)
        except:
            pass

    con.close()

    elapsed = time.time() - t0
    print(f"Feature extraction done in {elapsed:.1f}s. Total: {result.height:,} rows")
    if gt_set:
        print(f"  Positive labels: {int(result['label'].sum()):,}")

    del result; gc.collect()
    return elapsed


# ============================================================
# STEP 3: TRAIN LightGBM WITH HARD NEGATIVES
# ============================================================
FEATURE_COLS = [
    "exact_name_match", "exact_legal_match", "sorted_name_match",
    "exact_address_match", "country_match", "addr_nums_match",
    "name_len_diff", "name_len_ratio", "addr_len_diff",
    "retrieved_by_name", "retrieved_by_addr", "retrieved_by_punct",
    "retrieved_by_token", "retrieved_by_bigram", "retrieved_by_phonetic",
    "retrieved_by_sorted", "retrieved_by_postal", "num_channels",
    "name_jaccard", "name_containment", "name_shared_tokens",
    "addr_jaccard", "addr_containment", "addr_shared_tokens",
    "name_fuzz_ratio", "name_partial_ratio", "name_jaro_winkler",
    "addr_fuzz_ratio",
]


def train_lgbm(train_features_path, val_features_path, model_output_path, metrics_path):
    """Train LightGBM with hard negative sampling."""
    print("=" * 60)
    print("LIGHTGBM TRAINING WITH HARD NEGATIVES")
    print("=" * 60)

    df_train = pl.read_parquet(train_features_path)
    df_val = pl.read_parquet(val_features_path)

    print(f"  Train: {df_train.height:,} rows")
    print(f"  Val: {df_val.height:,} rows")

    # Separate positives and negatives
    pos_train = df_train.filter(pl.col("label") == 1)
    neg_train = df_train.filter(pl.col("label") == 0)

    print(f"  Train positives: {pos_train.height:,}")
    print(f"  Train negatives: {neg_train.height:,}")

    # Hard negative mining: select negatives that are most confusing
    # Negatives with high name similarity or high address similarity
    hard_neg = neg_train.filter(
        (pl.col("name_fuzz_ratio") > 0.3) |
        (pl.col("name_jaccard") > 0.2) |
        (pl.col("exact_address_match") == 1) |
        (pl.col("addr_jaccard") > 0.3) |
        (pl.col("num_channels") >= 2)
    )
    print(f"  Hard negatives (pre-sample): {hard_neg.height:,}")

    # Random negatives (smaller sample for diversity)
    max_random_neg = min(pos_train.height * 2, neg_train.height)
    random_neg = neg_train.sample(n=min(max_random_neg, neg_train.height), seed=42)

    # Combine: all positives + hard negatives + random negatives
    train_balanced = pl.concat([pos_train, hard_neg, random_neg]).unique(
        subset=["source1_entity_id", "candidate_entity_id"]
    )
    print(f"  Training set after hard neg mining: {train_balanced.height:,}")
    print(f"    Positives: {train_balanced.filter(pl.col('label')==1).height:,}")
    print(f"    Negatives: {train_balanced.filter(pl.col('label')==0).height:,}")

    # Prepare data
    X_train = train_balanced.select(FEATURE_COLS).to_pandas()
    y_train = train_balanced["label"].to_pandas()

    X_val = df_val.select(FEATURE_COLS).to_pandas()
    y_val = df_val["label"].to_pandas()

    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }

    print("Training LightGBM...")
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(50),
        ]
    )
    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s, best iteration: {model.best_iteration}")

    # Save model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    imp_dict = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    print("Feature importance (gain):")
    for fname, fval in imp_dict[:10]:
        print(f"    {fname}: {fval:.0f}")

    metrics = {
        "best_iteration": model.best_iteration,
        "train_time": train_time,
        "feature_importance": {f: float(i) for f, i in imp_dict},
        "train_size": train_balanced.height,
        "pos_count": int(pos_count),
        "neg_count": int(neg_count),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    del df_train, df_val, pos_train, neg_train, hard_neg, random_neg, train_balanced
    gc.collect()

    return model


# ============================================================
# STEP 4: PREDICT IN CHUNKS
# ============================================================
def predict_chunks(model_path, features_path, output_path, chunk_size=2_000_000):
    """Predict in chunks to limit memory."""
    print("=" * 60)
    print("PREDICTION (chunked)")
    print("=" * 60)

    model = joblib.load(model_path)
    total = pl.scan_parquet(features_path).select(pl.len()).collect().item()
    print(f"  Total pairs: {total:,}")

    all_preds = []
    for offset in range(0, total, chunk_size):
        df = pl.scan_parquet(features_path).slice(offset, chunk_size).collect()
        X = df.select(FEATURE_COLS).to_pandas()
        preds = model.predict(X)

        result = df.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
            pl.Series("probability", preds.astype(np.float32))
        )
        all_preds.append(result)
        print(f"  Predicted chunk {offset:,}-{offset+df.height:,}")
        del df, X, preds, result
        gc.collect()

    final = pl.concat(all_preds)
    final.write_parquet(output_path)
    print(f"  Saved predictions: {final.height:,} rows")
    del all_preds, final; gc.collect()


# ============================================================
# STEP 5: THRESHOLD-OPTIMIZED 0/1/MANY DECISION
# ============================================================
def optimize_threshold(predictions_path, gt_path):
    """Find optimal threshold for Macro F0.5."""
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)

    preds = pl.read_parquet(predictions_path)
    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')

    # Build GT mapping
    gt_map = {}
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matched = set(m.strip() for m in row['matched_entity_ids'].split(',')) if row['matched_entity_ids'].strip() else set()
        gt_map[s1] = matched

    def compute_macro_f05(threshold):
        matched_above = preds.filter(pl.col("probability") >= threshold)
        pred_map = {}
        for row in matched_above.iter_rows():
            s1, cand, prob = row
            if s1 not in pred_map:
                pred_map[s1] = set()
            pred_map[s1].add(cand)

        f05_scores = []
        for s1, gt_set in gt_map.items():
            pred_set = pred_map.get(s1, set())
            if not gt_set and not pred_set:
                f05_scores.append(1.0)
            elif not gt_set and pred_set:
                f05_scores.append(0.0)
            elif gt_set and not pred_set:
                f05_scores.append(0.0)
            else:
                tp = len(gt_set & pred_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if p + r == 0:
                    f05_scores.append(0.0)
                else:
                    f05_scores.append((1.25 * p * r) / (0.25 * p + r))

        return np.mean(f05_scores)

    # Grid search over thresholds
    best_f05 = 0
    best_threshold = 0.5
    thresholds = np.arange(0.05, 0.95, 0.05).tolist()
    thresholds += np.arange(0.01, 0.15, 0.01).tolist()  # Fine-grained low thresholds
    thresholds = sorted(set(thresholds))

    print(f"  Testing {len(thresholds)} thresholds...")
    for t in thresholds:
        f05 = compute_macro_f05(t)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
        if abs(t - 0.5) < 0.001 or f05 > best_f05 - 0.01:
            print(f"    threshold={t:.3f}: Macro F0.5={f05:.4f}")

    # Fine-grained search around best
    fine_range = np.arange(max(0.01, best_threshold - 0.05),
                           min(0.99, best_threshold + 0.05), 0.005)
    for t in fine_range:
        f05 = compute_macro_f05(t)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t

    print(f"  BEST threshold: {best_threshold:.4f}, Macro F0.5: {best_f05:.4f}")
    return best_threshold, best_f05


def generate_decisions(predictions_path, s1_path, output_path, candidate_out_path,
                       threshold=0.5):
    """Generate final matching and candidate TSV files."""
    print("=" * 60)
    print("DECISION ENGINE (0/1/MANY)")
    print("=" * 60)

    s1 = pl.read_parquet(s1_path)
    all_s1 = s1.select(["entity_id"]).rename({"entity_id": "source1_entity_id"})

    # Candidate pairs export
    print("  Generating candidate_pairs.tsv...")
    cands = pl.scan_parquet(predictions_path).select([
        "source1_entity_id", "candidate_entity_id"
    ]).group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("candidates")
    ).with_columns(
        pl.col("candidates").list.join(",")
    ).collect()

    final_cands = all_s1.join(cands, on="source1_entity_id", how="left").fill_null("")
    final_cands.select([
        "source1_entity_id",
        pl.col("candidates").alias("candidate_entity_ids")
    ]).write_csv(candidate_out_path, separator='\t', quote_style='never')

    # Matching results export
    print(f"  Generating matching_results.tsv (threshold={threshold:.4f})...")
    matches = pl.scan_parquet(predictions_path).filter(
        pl.col("probability") >= threshold
    ).select([
        "source1_entity_id", "candidate_entity_id"
    ]).group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("matches")
    ).with_columns(
        pl.col("matches").list.join(",")
    ).collect()

    final_matches = all_s1.join(matches, on="source1_entity_id", how="left").fill_null("")
    final_matches.select([
        "source1_entity_id",
        pl.col("matches").alias("matched_entity_ids")
    ]).write_csv(output_path, separator='\t', quote_style='never')

    # Stats
    zero_matches = final_matches.filter(pl.col("matches") == "").height
    all_with_matches = final_matches.filter(pl.col("matches") != "")
    singleton = all_with_matches.filter(~pl.col("matches").str.contains(",")).height
    multi = all_with_matches.filter(pl.col("matches").str.contains(",")).height

    print(f"  Total S1: {final_matches.height:,}")
    print(f"  Zero matches: {zero_matches:,}")
    print(f"  Single matches: {singleton:,}")
    print(f"  Multi matches: {multi:,}")

    return {"total": final_matches.height, "zero": zero_matches,
            "single": singleton, "multi": multi}


# ============================================================
# STEP 6: EVALUATION
# ============================================================
def evaluate_f05(gt_path, pred_path):
    """Calculate Macro F0.5."""
    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    pred = pd.read_csv(pred_path, sep='\t', dtype=str).fillna('')

    merged = pd.merge(gt, pred, on='source1_entity_id', how='left').fillna('')

    f05_scores = []
    precision_scores = []
    recall_scores = []

    for _, row in merged.iterrows():
        gt_str = row.get('matched_entity_ids_x', '')
        pred_str = row.get('matched_entity_ids_y', '')

        gt_set = set(x.strip() for x in gt_str.split(',') if x.strip()) if gt_str.strip() else set()
        pred_set = set(x.strip() for x in pred_str.split(',') if x.strip()) if pred_str.strip() else set()

        if not gt_set and not pred_set:
            p, r, f = 1.0, 1.0, 1.0
        elif not gt_set and pred_set:
            p, r, f = 0.0, 0.0, 0.0
        elif gt_set and not pred_set:
            p, r, f = 0.0, 0.0, 0.0
        else:
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if p + r == 0:
                f = 0.0
            else:
                f = (1.25 * p * r) / (0.25 * p + r)

        precision_scores.append(p)
        recall_scores.append(r)
        f05_scores.append(f)

    macro_p = np.mean(precision_scores)
    macro_r = np.mean(recall_scores)
    macro_f05 = np.mean(f05_scores)

    print(f"  Macro Precision: {macro_p:.4f}")
    print(f"  Macro Recall:    {macro_r:.4f}")
    print(f"  Macro F0.5:      {macro_f05:.4f}")

    return {"macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f05": float(macro_f05)}


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    overall_t0 = time.time()

    print("=" * 70)
    print("FINAL ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # Step 0: Ensure preprocessing is done
    print("\n--- STEP 0: Check Preprocessing ---")
    ensure_preprocessed()

    # Step 1: Retrieval V3 on train split
    print("\n--- STEP 1A: Retrieval V3 on TRAIN split ---")
    train_cands_path = ARTIFACTS_DIR / "train_candidates_v3.parquet"
    if not train_cands_path.exists():
        retrieval_v3(
            s1_path=PROCESSED_DIR / "train_s1_split.parquet",
            s2_path=PROCESSED_DIR / "train_s2.parquet",
            s3_path=PROCESSED_DIR / "train_s3.parquet",
            output_path=train_cands_path,
            gt_path=SPLITS_DIR / "train_gt_split.tsv",
            metrics_path=ARTIFACTS_DIR / "train_retrieval_v3_metrics.json"
        )
    else:
        print("  Train candidates V3 already exist, skipping.")

    print("\n--- STEP 1B: Retrieval V3 on VAL split ---")
    val_cands_path = ARTIFACTS_DIR / "val_candidates_v3.parquet"
    if not val_cands_path.exists():
        retrieval_v3(
            s1_path=PROCESSED_DIR / "val_s1_split.parquet",
            s2_path=PROCESSED_DIR / "train_s2.parquet",
            s3_path=PROCESSED_DIR / "train_s3.parquet",
            output_path=val_cands_path,
            gt_path=SPLITS_DIR / "val_gt_split.tsv",
            metrics_path=ARTIFACTS_DIR / "val_retrieval_v3_metrics.json"
        )
    else:
        print("  Val candidates V3 already exist, skipping.")

    # Step 2: Feature extraction
    print("\n--- STEP 2A: Features on TRAIN split ---")
    train_features_path = ARTIFACTS_DIR / "train_features_v3.parquet"
    if not train_features_path.exists():
        compute_features_duckdb(
            candidates_path=train_cands_path,
            s1_path=PROCESSED_DIR / "train_s1_split.parquet",
            s2_path=PROCESSED_DIR / "train_s2.parquet",
            s3_path=PROCESSED_DIR / "train_s3.parquet",
            output_path=train_features_path,
            gt_path=SPLITS_DIR / "train_gt_split.tsv",
            chunk_size=100_000
        )
    else:
        print("  Train features V3 already exist, skipping.")

    print("\n--- STEP 2B: Features on VAL split ---")
    val_features_path = ARTIFACTS_DIR / "val_features_v3.parquet"
    if not val_features_path.exists():
        compute_features_duckdb(
            candidates_path=val_cands_path,
            s1_path=PROCESSED_DIR / "val_s1_split.parquet",
            s2_path=PROCESSED_DIR / "train_s2.parquet",
            s3_path=PROCESSED_DIR / "train_s3.parquet",
            output_path=val_features_path,
            gt_path=SPLITS_DIR / "val_gt_split.tsv",
            chunk_size=100_000
        )
    else:
        print("  Val features V3 already exist, skipping.")

    # Step 3: Train LightGBM
    print("\n--- STEP 3: Train LightGBM ---")
    model_path = ARTIFACTS_DIR / "lgb_model_v3.pkl"
    model_metrics_path = ARTIFACTS_DIR / "model_metrics_v3.json"
    model = train_lgbm(train_features_path, val_features_path,
                       model_path, model_metrics_path)

    # Step 4: Predict on validation
    print("\n--- STEP 4: Predict on Validation ---")
    val_preds_path = ARTIFACTS_DIR / "val_predictions_v3.parquet"
    predict_chunks(model_path, val_features_path, val_preds_path)

    # Step 5: Optimize threshold
    print("\n--- STEP 5: Optimize Threshold ---")
    best_threshold, best_f05 = optimize_threshold(
        val_preds_path, SPLITS_DIR / "val_gt_split.tsv"
    )

    # Step 6: Generate validation decisions and evaluate
    print("\n--- STEP 6: Validation Decision & Evaluation ---")
    val_match_path = ARTIFACTS_DIR / "val_matching_results_v3.tsv"
    val_cand_tsv_path = ARTIFACTS_DIR / "val_candidate_pairs_v3.tsv"
    dec_stats = generate_decisions(
        val_preds_path,
        PROCESSED_DIR / "val_s1_split.parquet",
        val_match_path,
        val_cand_tsv_path,
        threshold=best_threshold
    )

    val_metrics = evaluate_f05(
        SPLITS_DIR / "val_gt_split.tsv",
        val_match_path
    )

    # Save validation metrics
    with open(ARTIFACTS_DIR / "val_f05_metrics_v3.json", 'w') as f:
        json.dump({**val_metrics, "threshold": best_threshold}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: Macro F0.5 = {val_metrics['macro_f05']:.4f}")
    print(f"  Precision: {val_metrics['macro_precision']:.4f}")
    print(f"  Recall: {val_metrics['macro_recall']:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"{'='*60}")

    # ============================================================
    # STEP 7: FULL TEST INFERENCE
    # ============================================================
    print("\n--- STEP 7A: Preprocess test data ---")
    ensure_preprocessed()

    print("\n--- STEP 7B: Test Retrieval ---")
    test_cands_path = ARTIFACTS_DIR / "test_candidates_v3.parquet"
    if not test_cands_path.exists():
        retrieval_v3(
            s1_path=PROCESSED_DIR / "test_s1.parquet",
            s2_path=PROCESSED_DIR / "test_s2.parquet",
            s3_path=PROCESSED_DIR / "test_s3.parquet",
            output_path=test_cands_path
        )
    else:
        print("  Test candidates V3 already exist, skipping.")

    print("\n--- STEP 7C: Test Features ---")
    test_features_path = ARTIFACTS_DIR / "test_features_v3.parquet"
    if not test_features_path.exists():
        compute_features_duckdb(
            candidates_path=test_cands_path,
            s1_path=PROCESSED_DIR / "test_s1.parquet",
            s2_path=PROCESSED_DIR / "test_s2.parquet",
            s3_path=PROCESSED_DIR / "test_s3.parquet",
            output_path=test_features_path,
            chunk_size=100_000
        )
    else:
        print("  Test features V3 already exist, skipping.")

    print("\n--- STEP 7D: Test Prediction ---")
    test_preds_path = ARTIFACTS_DIR / "test_predictions_v3.parquet"
    predict_chunks(model_path, test_features_path, test_preds_path)

    print("\n--- STEP 7E: Test Decision ---")
    test_stats = generate_decisions(
        test_preds_path,
        PROCESSED_DIR / "test_s1.parquet",
        OUTPUT_DIR / "matching_results.tsv",
        OUTPUT_DIR / "candidate_pairs.tsv",
        threshold=best_threshold
    )

    # ============================================================
    # STEP 8: FINAL SUMMARY
    # ============================================================
    total_time = time.time() - overall_t0

    print(f"\n{'='*70}")
    print("FINAL MODEL")
    print(f"Configuration: Retrieval V3 (9 channels) + LightGBM ({len(FEATURE_COLS)} features) + Hard Negatives + Threshold Optimization")
    print(f"Validation Macro F0.5: {val_metrics['macro_f05']:.4f}")
    print(f"Validation Precision:  {val_metrics['macro_precision']:.4f}")
    print(f"Validation Recall:     {val_metrics['macro_recall']:.4f}")
    print(f"Best Threshold:        {best_threshold:.4f}")
    print()
    print("TEST")
    print(f"Test S1 Count:    {test_stats['total']:,}")
    print(f"Output Rows:      {test_stats['total']:,}")
    print(f"Predicted Zero:   {test_stats['zero']:,}")
    print(f"Predicted One:    {test_stats['single']:,}")
    print(f"Predicted Many:   {test_stats['multi']:,}")
    print()
    print("FILES")
    print(f"matching_results.tsv: {OUTPUT_DIR / 'matching_results.tsv'}")
    print(f"candidate_pairs.tsv:  {OUTPUT_DIR / 'candidate_pairs.tsv'}")
    print()
    print(f"Total Runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*70}")

    # Save final metrics
    final_metrics = {
        "validation": val_metrics,
        "threshold": best_threshold,
        "test_stats": test_stats,
        "total_runtime_seconds": total_time,
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
    }
    with open(ARTIFACTS_DIR / "final_metrics_v3.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()

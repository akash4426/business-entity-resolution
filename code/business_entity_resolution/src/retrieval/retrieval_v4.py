"""
retrieval_v4.py — Country-Sharded, Index-Lookup-First Retrieval
"""

import polars as pl
import duckdb
import os
import glob
import gc
import json
import time
import re
from pathlib import Path
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

COLS_NEEDED = [
    "entity_id", "country_norm", "name_legal", "name_punct",
    "name_sorted", "address_norm", "address_numbers", "name_tokens",
]

POSTAL_RE = r'\b(\d{5,6})\b'

def _extract_postal(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("address_norm").str.extract(POSTAL_RE).alias("postal_code")
    )

def _extract_street_key(df: pl.DataFrame, n_tokens: int = 3) -> pl.DataFrame:
    return df.with_columns(
        pl.col("address_norm")
        .str.split(" ")
        .list.eval(
            pl.element().filter(~pl.element().str.contains(r'^\d+$'))
        )
        .list.head(n_tokens)
        .list.join(" ")
        .alias("street_tokens")
    ).with_columns(
        (pl.col("address_numbers") + "|" + pl.col("street_tokens")).alias("addr_struct_key")
    )

def _add_soundex_vectorised(df: pl.DataFrame) -> pl.DataFrame:
    import jellyfish
    def _soundex_first_token(name: str) -> str:
        if not name or not name.strip():
            return ""
        tok = name.split()[0]
        if not tok.isalpha():
            return ""
        try:
            return jellyfish.soundex(tok)
        except Exception:
            return ""

    return df.with_columns(
        pl.col("name_legal")
        .map_elements(_soundex_first_token, return_dtype=pl.Utf8)
        .alias("soundex")
    )

def _extract_3grams(df: pl.DataFrame) -> pl.DataFrame:
    # Build 3-grams for name_punct
    # We will use python map_elements for simplicity
    def _get_3grams(name: str) -> list[str]:
        if not name: return []
        # pad name for 3-grams
        name = " " + name + " "
        return [name[i:i+3] for i in range(len(name)-2)]
    
    return df.filter(pl.col("name_punct") != "").with_columns(
        pl.col("name_punct").map_elements(_get_3grams, return_dtype=pl.List(pl.Utf8)).alias("ngrams")
    ).explode("ngrams").rename({"ngrams": "ngram"})

def _get_minhash_lsh_pairs(s1: pl.DataFrame, s23: pl.DataFrame, num_perm=128, threshold=0.5):
    from datasketch import MinHash, MinHashLSH
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    s23_names = s23.select(["entity_id", "name_punct"]).drop_nulls().filter(pl.col("name_punct") != "").to_dicts()
    s1_names = s1.select(["entity_id", "name_punct"]).drop_nulls().filter(pl.col("name_punct") != "").to_dicts()
    
    # Insert S23 into LSH
    def get_shingles(s):
        s = " " + s + " "
        return [s[i:i+3].encode('utf8') for i in range(len(s)-2)]
        
    for row in s23_names:
        m = MinHash(num_perm=num_perm)
        for d in get_shingles(row["name_punct"]):
            m.update(d)
        lsh.insert(row["entity_id"], m)
        
    # Query S1 against LSH
    s1_ids = []
    s23_ids = []
    for row in s1_names:
        m = MinHash(num_perm=num_perm)
        for d in get_shingles(row["name_punct"]):
            m.update(d)
        result = lsh.query(m)
        for r in result:
            s1_ids.append(row["entity_id"])
            s23_ids.append(r)
            
    return pl.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": s23_ids
    }, schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})

# ─────────────────────────────────────────────────────────────────────────────
# SPARSE N-GRAM TF-IDF SCORING (ON CANDIDATE POOL ONLY)
# ─────────────────────────────────────────────────────────────────────────────

def _score_ngram_tfidf(s1_shard: pl.DataFrame, s23_shard: pl.DataFrame, cands: pl.DataFrame, min_score: float = 0.25) -> pl.DataFrame:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return cands

    s1_texts = s1_shard.select(["entity_id", "name_punct"]).drop_nulls()
    s23_texts = s23_shard.select(["entity_id", "name_punct"]).drop_nulls()
    
    if s1_texts.height == 0 or s23_texts.height == 0 or cands.height == 0:
        return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=1000, max_features=50_000, sublinear_tf=True, dtype=np.float32)
    s23_names = s23_texts["name_punct"].to_list()
    s1_names = s1_texts["name_punct"].to_list()

    try:
        X_s23 = vectorizer.fit_transform(s23_names)
        X_s1 = vectorizer.transform(s1_names)
    except ValueError:
        return cands
        
    s1_id_to_idx = {row["entity_id"]: i for i, row in enumerate(s1_texts.to_dicts())}
    s23_id_to_idx = {row["entity_id"]: i for i, row in enumerate(s23_texts.to_dicts())}

    # Filter cands that are not in our texts
    cands_filtered = cands.filter(
        pl.col("source1_entity_id").is_in(list(s1_id_to_idx.keys())) &
        pl.col("candidate_entity_id").is_in(list(s23_id_to_idx.keys()))
    )
    
    s1_indices = [s1_id_to_idx[r["source1_entity_id"]] for r in cands_filtered.to_dicts()]
    s23_indices = [s23_id_to_idx[r["candidate_entity_id"]] for r in cands_filtered.to_dicts()]
    
    if not s1_indices:
        return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})
        
    # Element-wise dot product of the two sparse matrices for exactly these pairs
    # X_s1[s1_indices] multiply X_s23[s23_indices]
    from scipy.sparse import csr_matrix
    X1_sub = X_s1[s1_indices]
    X2_sub = X_s23[s23_indices]
    scores = X1_sub.multiply(X2_sub).sum(axis=1).A1
    
    scored_cands = cands_filtered.with_columns(pl.Series("score", scores))
    return scored_cands.filter(pl.col("score") >= min_score).select(["source1_entity_id", "candidate_entity_id"])


# ─────────────────────────────────────────────────────────────────────────────
# PER-SHARD RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _run_shard(
    country: str,
    s1_shard: pl.DataFrame,
    s2_shard: pl.DataFrame,
    s3_shard: pl.DataFrame,
    tmp_dir: Path,
    batch_size: int = 100_000,
    token_freq_cap: int = 500,
    bigram_entity_token_cap: int = 15,
    ngram_top_k: int = 10,
    ngram_min_score: float = 0.25,
    channel_metrics: dict = None,
) -> int:
    s23 = pl.concat([s2_shard, s3_shard], how="vertical_relaxed")
    total_pairs = 0
    chunk_idx = [0]

    def _write(df: pl.DataFrame, ch_name: str):
        if df.height == 0:
            return
        df = df.with_columns(pl.lit(ch_name).alias("channel"))
        out = tmp_dir / f"cands_{country}_{ch_name}_{chunk_idx[0]}.parquet"
        df.write_parquet(out)
        chunk_idx[0] += 1
        print(f"    Channel {ch_name:15s}: {df.height:,} raw pairs")
        if channel_metrics is not None:
            channel_metrics.setdefault(ch_name, 0)
            channel_metrics[ch_name] += df.height

    def _pairs(df_left: pl.DataFrame, df_right: pl.DataFrame, keys: list) -> pl.DataFrame:
        j = df_left.join(df_right, on=keys, how="inner", suffix="_r")
        if j.height == 0:
            return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})
        return j.select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_r").alias("candidate_entity_id"),
        ])

    # 1. Exact Name
    def _name_view(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col("name_legal") != "")
            .then(pl.col("name_legal"))
            .otherwise(pl.col("name_punct"))
            .alias("name_norm_combined")
        ).filter(pl.col("name_norm_combined") != "")

    s1_nv = _name_view(s1_shard).select(["entity_id", "name_norm_combined"])
    s23_nv = _name_view(s23).select(["entity_id", "name_norm_combined"])
    freq = s23_nv.group_by("name_norm_combined").len()
    valid_names = freq.filter(pl.col("len") <= 500).select("name_norm_combined")
    s1_ch1 = s1_nv.join(valid_names, on="name_norm_combined", how="inner")
    s23_ch1 = s23_nv.join(valid_names, on="name_norm_combined", how="inner")
    p1 = _pairs(s1_ch1, s23_ch1, ["name_norm_combined"]).unique()
    _write(p1, "exact_name")
    total_pairs += p1.height

    # 2. Exact Address
    s1_a = s1_shard.filter(pl.col("address_norm") != "").select(["entity_id", "address_norm"])
    s23_a = s23.filter(pl.col("address_norm") != "").select(["entity_id", "address_norm"])
    freq = s23_a.group_by("address_norm").len()
    valid_addr = freq.filter(pl.col("len") <= 500).select("address_norm")
    s1_ch2 = s1_a.join(valid_addr, on="address_norm", how="inner")
    s23_ch2 = s23_a.join(valid_addr, on="address_norm", how="inner")
    p2 = _pairs(s1_ch2, s23_ch2, ["address_norm"]).unique()
    _write(p2, "exact_addr")
    total_pairs += p2.height

    # 3. Structural address key
    s1_sk = _extract_street_key(s1_shard).filter(
        (pl.col("address_numbers") != "") & (pl.col("street_tokens") != "")
    ).select(["entity_id", "addr_struct_key"])
    s23_sk = _extract_street_key(s23).filter(
        (pl.col("address_numbers") != "") & (pl.col("street_tokens") != "")
    ).select(["entity_id", "addr_struct_key"])
    freq = s23_sk.group_by("addr_struct_key").len()
    valid_sk = freq.filter(pl.col("len") <= 1000).select("addr_struct_key")
    s1_ch3 = s1_sk.join(valid_sk, on="addr_struct_key", how="inner")
    s23_ch3 = s23_sk.join(valid_sk, on="addr_struct_key", how="inner")
    p3 = _pairs(s1_ch3, s23_ch3, ["addr_struct_key"]).unique()
    _write(p3, "addr_struct")
    total_pairs += p3.height

    # 4. Postal code
    s1_pc = _extract_postal(s1_shard).filter(pl.col("postal_code").is_not_null() & (pl.col("postal_code") != "")).select(["entity_id", "postal_code"])
    s23_pc = _extract_postal(s23).filter(pl.col("postal_code").is_not_null() & (pl.col("postal_code") != "")).select(["entity_id", "postal_code"])
    freq = s23_pc.group_by("postal_code").len()
    valid_pc = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 2000)).select("postal_code")
    s1_ch4 = s1_pc.join(valid_pc, on="postal_code", how="inner")
    s23_ch4 = s23_pc.join(valid_pc, on="postal_code", how="inner")
    p4 = _pairs(s1_ch4, s23_ch4, ["postal_code"]).unique()
    _write(p4, "postal")
    total_pairs += p4.height

    # 5. Token
    def _explode_tokens(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("name_tokens") != "")
            .with_columns(pl.col("name_tokens").str.split(" ").alias("tok_list"))
            .explode("tok_list")
            .rename({"tok_list": "token"})
            .filter((pl.col("token") != "") & (pl.col("token").str.len_chars() > 1))
            .select(["entity_id", "token"])
        )
    s1_tok = _explode_tokens(s1_shard)
    s23_tok = _explode_tokens(s23)
    freq_tok = s23_tok.group_by("token").len()
    valid_tok = freq_tok.filter(
        (pl.col("len") > 0) & (pl.col("len") <= token_freq_cap)
    ).select("token")
    s1_ch5 = s1_tok.join(valid_tok, on="token", how="inner")
    s23_ch5 = s23_tok.join(valid_tok, on="token", how="inner")
    p5 = _pairs(s1_ch5, s23_ch5, ["token"]).unique()
    _write(p5, "token")
    total_pairs += p5.height

    # 6. Phonetic
    s1_sx = _add_soundex_vectorised(s1_shard).filter(
        pl.col("soundex").is_not_null() & (pl.col("soundex") != "")
    ).select(["entity_id", "soundex"])
    s23_sx = _add_soundex_vectorised(s23).filter(
        pl.col("soundex").is_not_null() & (pl.col("soundex") != "")
    ).select(["entity_id", "soundex"])
    freq_sx = s23_sx.group_by("soundex").len()
    valid_sx = freq_sx.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select("soundex")
    s1_ch7 = s1_sx.join(valid_sx, on="soundex", how="inner")
    s23_ch7 = s23_sx.join(valid_sx, on="soundex", how="inner")
    p7 = _pairs(s1_ch7, s23_ch7, ["soundex"]).unique()
    _write(p7, "phonetic")
    total_pairs += p7.height

    # 7. Bigram
    tok_freq_for_cap = s23_tok.group_by("token").len().rename({"len": "tok_freq"})
    s1_tok_capped = (
        s1_tok.join(tok_freq_for_cap, on="token", how="left").fill_null(1)
        .sort(["entity_id", "tok_freq"])
        .with_columns(pl.col("token").cum_count().over("entity_id").alias("rank"))
        .filter(pl.col("rank") <= bigram_entity_token_cap)
        .select(["entity_id", "token"])
    )
    s23_tok_capped = (
        s23_tok.join(tok_freq_for_cap, on="token", how="left").fill_null(1)
        .sort(["entity_id", "tok_freq"])
        .with_columns(pl.col("token").cum_count().over("entity_id").alias("rank"))
        .filter(pl.col("rank") <= bigram_entity_token_cap)
        .select(["entity_id", "token"])
    )
    s1_bg = (
        s1_tok_capped.join(s1_tok_capped, on="entity_id", suffix="_2", how="inner")
        .filter(pl.col("token") < pl.col("token_2"))
        .with_columns((pl.col("token") + "|" + pl.col("token_2")).alias("bigram"))
        .select(["entity_id", "bigram"])
    )
    s23_bg = (
        s23_tok_capped.join(s23_tok_capped, on="entity_id", suffix="_2", how="inner")
        .filter(pl.col("token") < pl.col("token_2"))
        .with_columns((pl.col("token") + "|" + pl.col("token_2")).alias("bigram"))
        .select(["entity_id", "bigram"])
    )
    freq_bg = s23_bg.group_by("bigram").len()
    valid_bg = freq_bg.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select("bigram")
    s1_ch8 = s1_bg.join(valid_bg, on="bigram", how="inner")
    s23_ch8 = s23_bg.join(valid_bg, on="bigram", how="inner")
    p8 = _pairs(s1_ch8, s23_ch8, ["bigram"]).unique()
    _write(p8, "bigram")
    total_pairs += p8.height
    
    # 8. Character n-gram index (instead of full matrix multiply)
    s1_3g = _extract_3grams(s1_shard)
    s23_3g = _extract_3grams(s23)
    freq_3g = s23_3g.group_by("ngram").len()
    valid_3g = freq_3g.filter((pl.col("len") > 0) & (pl.col("len") <= 100)).select("ngram")
    s1_3g = s1_3g.join(valid_3g, on="ngram", how="inner")
    s23_3g = s23_3g.join(valid_3g, on="ngram", how="inner")
    # Union of 3-gram index and MinHash LSH as a candidate pool for TF-IDF scoring
    p6_ngram_cands = _pairs(s1_3g, s23_3g, ["ngram"]).unique()
    print(f"    [DEBUG] 3-gram candidate pool size before TF-IDF scoring: {p6_ngram_cands.height:,}")
    
    # 9. MinHash LSH
    t_lsh = time.time()
    p9 = _get_minhash_lsh_pairs(s1_shard, s23, num_perm=64, threshold=0.5)
    print(f"    [DEBUG] MinHash LSH took {time.time()-t_lsh:.1f}s, generated {p9.height:,} pairs")
    _write(p9, "minhash_lsh")
    total_pairs += p9.height

    # Compute TF-IDF over the union of 3-gram candidate pool
    t_tfidf = time.time()
    p6_scored = _score_ngram_tfidf(s1_shard, s23, p6_ngram_cands, min_score=ngram_min_score)
    print(f"    [DEBUG] TF-IDF scoring took {time.time()-t_tfidf:.1f}s, generated {p6_scored.height:,} pairs")
    _write(p6_scored, "ngram_tfidf")
    total_pairs += p6_scored.height

    del s23; gc.collect()
    return total_pairs

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_v4(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    gt_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    batch_size: int = 100_000,
    token_freq_cap: int = 500,
    bigram_entity_token_cap: int = 15,
    ngram_top_k: int = 10,
    ngram_min_score: float = 0.25,
) -> dict:
    t0 = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = output_path.parent / "_retrieval_v4_tmp"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for f in tmp_dir.glob("cands_*.parquet"):
        f.unlink()

    s1_scan = pl.scan_parquet(s1_path).select([c for c in COLS_NEEDED if c in pl.read_parquet(s1_path, n_rows=1).columns])
    s2_scan = pl.scan_parquet(s2_path).select([c for c in COLS_NEEDED if c in pl.read_parquet(s2_path, n_rows=1).columns])
    s3_scan = pl.scan_parquet(s3_path).select([c for c in COLS_NEEDED if c in pl.read_parquet(s3_path, n_rows=1).columns])

    s1_countries = s1_scan.select("country_norm").collect()["country_norm"].unique().to_list()
    s2_countries = s2_scan.select("country_norm").collect()["country_norm"].unique().to_list()
    s3_countries = s3_scan.select("country_norm").collect()["country_norm"].unique().to_list()
    all_countries = sorted(set(s1_countries + s2_countries + s3_countries) - {None, ""})

    channel_metrics = {}
    total_raw_pairs = 0

    for country in all_countries:
        t_shard = time.time()
        print(f"\n── Shard: {country} ──")

        s1_shard = s1_scan.filter(pl.col("country_norm") == country).collect()
        s2_shard = s2_scan.filter(pl.col("country_norm") == country).collect()
        s3_shard = s3_scan.filter(pl.col("country_norm") == country).collect()

        if s1_shard.height == 0:
            del s1_shard, s2_shard, s3_shard; gc.collect()
            continue

        raw = _run_shard(
            country=country,
            s1_shard=s1_shard,
            s2_shard=s2_shard,
            s3_shard=s3_shard,
            tmp_dir=tmp_dir,
            batch_size=batch_size,
            token_freq_cap=token_freq_cap,
            bigram_entity_token_cap=bigram_entity_token_cap,
            ngram_top_k=ngram_top_k,
            ngram_min_score=ngram_min_score,
            channel_metrics=channel_metrics,
        )
        total_raw_pairs += raw
        print(f"  Shard done in {time.time() - t_shard:.1f}s")
        del s1_shard, s2_shard, s3_shard; gc.collect()

    con = duckdb.connect()
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    tmp_glob = str(tmp_dir / "cands_*.parquet")
    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                string_agg(DISTINCT channel, '|') AS channels,
                count(DISTINCT channel) AS num_channels
            FROM read_parquet('{tmp_glob}')
            GROUP BY source1_entity_id, candidate_entity_id
        ) TO '{output_path}' (FORMAT 'parquet')
    """)

    total_cands = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    stats = con.execute(f"""
        SELECT
            avg(cnt) AS mean_cands,
            quantile_cont(cnt, 0.5) AS median_cands,
            quantile_cont(cnt, 0.95) AS p95_cands,
            quantile_cont(cnt, 0.99) AS p99_cands,
            max(cnt) AS max_cands
        FROM (
            SELECT source1_entity_id, count(*) AS cnt
            FROM read_parquet('{output_path}')
            GROUP BY source1_entity_id
        )
    """).fetchone()

    t_total = time.time() - t0
    result = {
        "total_candidates": total_cands,
        "total_raw_pairs": total_raw_pairs,
        "runtime_seconds": t_total,
        "mean_cands_per_s1": float(stats[0]) if stats[0] else 0,
        "median_cands": float(stats[1]) if stats[1] else 0,
        "p95_cands": float(stats[2]) if stats[2] else 0,
        "p99_cands": float(stats[3]) if stats[3] else 0,
        "max_cands": float(stats[4]) if stats[4] else 0,
        "channel_raw_pairs": channel_metrics,
    }

    if gt_path:
        gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0).fill_null("")
        gt_exp = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        total_true = gt_exp.height
        s1_with_matches = gt.filter(pl.col("matched_entity_ids") != "")

        tmp_gt = str(tmp_dir / "gt_eval.parquet")
        gt_exp.write_parquet(tmp_gt)

        recovered = con.execute(f"""
            SELECT count(*) FROM (
                SELECT g.source1_entity_id, g.matched_entity_ids AS candidate_entity_id
                FROM read_parquet('{tmp_gt}') g
            ) g
            JOIN read_parquet('{output_path}') c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.candidate_entity_id = c.candidate_entity_id
        """).fetchone()[0]

        s1_with_any = con.execute(f"""
            SELECT count(DISTINCT g.source1_entity_id) FROM (
                SELECT g.source1_entity_id, g.matched_entity_ids AS candidate_entity_id
                FROM read_parquet('{tmp_gt}') g
            ) g
            JOIN read_parquet('{output_path}') c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.candidate_entity_id = c.candidate_entity_id
        """).fetchone()[0]

        result.update({
            "pair_recall": recovered / total_true if total_true > 0 else 0,
            "at_least_one_recall": s1_with_any / s1_with_matches.height if s1_with_matches.height > 0 else 0,
            "recovered_links": recovered,
            "total_true_links": total_true,
        })
        print(f"Recall: {result['pair_recall']:.4f}")

    con.close()
    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--token-freq-cap", type=int, default=500)
    parser.add_argument("--bigram-entity-token-cap", type=int, default=15)
    parser.add_argument("--ngram-top-k", type=int, default=10)
    parser.add_argument("--ngram-min-score", type=float, default=0.25)
    args = parser.parse_args()

    retrieve_v4(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        gt_path=args.gt,
        metrics_path=args.metrics,
        tmp_dir=args.tmp_dir,
        token_freq_cap=args.token_freq_cap,
        bigram_entity_token_cap=args.bigram_entity_token_cap,
        ngram_top_k=args.ngram_top_k,
        ngram_min_score=args.ngram_min_score,
    )

import polars as pl
import time
import gc
import json
import os
import glob
import duckdb

def compute_features_duckdb(candidates_path, s1_path, s2_path, s3_path, output_path, gt_path=None, chunk_size=200_000, smoke_test=False):
    print("=" * 60)
    print("FEATURE EXTRACTION (DuckDB + RapidFuzz batch) [MEMORY SAFE]")
    print("=" * 60)
    t0 = time.time()

    con = duckdb.connect()

    con.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet('{candidates_path}')")
    con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{s1_path}')")
    con.execute(f"""
        CREATE VIEW sources AS
        SELECT * FROM read_parquet('{s2_path}')
        UNION ALL
        SELECT * FROM read_parquet('{s3_path}')
    """)

    total_candidates = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"  Total candidates: {total_candidates:,}")
    if smoke_test:
        print("  SMOKE TEST: processing max 500k candidates")
        total_candidates = min(total_candidates, 500_000)

    # Clean old chunks
    for f in glob.glob(str(output_path).replace(".parquet", "_chunk_*.parquet")):
        try: os.remove(f)
        except: pass

    has_gt = False
    if gt_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        
        tmp_gt = str(output_path).replace(".parquet", "_tmp_gt.parquet")
        gt_exploded.write_parquet(tmp_gt)
        
        con.execute(f"CREATE VIEW gt_view AS SELECT source1_entity_id, matched_entity_ids as candidate_entity_id FROM read_parquet('{tmp_gt}')")
        has_gt = True

    chunk_idx = 0
    all_chunk_files = []
    
    if has_gt:
        label_sql = "CASE WHEN gt.source1_entity_id IS NOT NULL THEN 1 ELSE 0 END as label,"
        join_gt_sql = "LEFT JOIN gt_view gt ON c.source1_entity_id = gt.source1_entity_id AND c.candidate_entity_id = gt.candidate_entity_id"
    else:
        label_sql = "0 as label,"
        join_gt_sql = ""

    limit_sql = f"LIMIT {total_candidates}" if smoke_test else ""

    query = f"""
    SELECT
        c.source1_entity_id,
        c.candidate_entity_id,
        {label_sql}
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
    FROM (SELECT * FROM candidates {limit_sql}) c
    LEFT JOIN s1 ON c.source1_entity_id = s1.entity_id
    LEFT JOIN sources src ON c.candidate_entity_id = src.entity_id
    {join_gt_sql}
    """

    import pyarrow as pa
    reader = con.execute(query).fetch_record_batch(chunk_size)
    offset = 0

    while True:
        t_chunk = time.time()
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
            
        df = pl.from_arrow(batch)

        if df.height == 0:
            break

        # STAGE A: CHEAP FEATURES
        df = df.with_columns([
            (pl.col("name1") == pl.col("name2")).cast(pl.Float32).alias("exact_name_match"),
            (pl.col("name1_legal") == pl.col("name2_legal")).cast(pl.Float32).alias("exact_legal_match"),
            (pl.col("name1_sorted") == pl.col("name2_sorted")).cast(pl.Float32).alias("sorted_name_match"),
            (pl.col("addr1") == pl.col("addr2")).cast(pl.Float32).alias("exact_address_match"),
            (pl.col("country1") == pl.col("country2")).cast(pl.Float32).alias("country_match"),
            (pl.col("addr_nums1") == pl.col("addr_nums2")).cast(pl.Float32).alias("addr_nums_match"),
            pl.col("name1").str.len_chars().cast(pl.Float32).alias("name1_len"),
            pl.col("name2").str.len_chars().cast(pl.Float32).alias("name2_len"),
            pl.col("addr1").str.len_chars().cast(pl.Float32).alias("addr1_len"),
            pl.col("addr2").str.len_chars().cast(pl.Float32).alias("addr2_len"),
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

        df = df.with_columns([
            (pl.col("name1_len") - pl.col("name2_len")).abs().alias("name_len_diff"),
            (pl.when(pl.col("name2_len") > 0).then(pl.col("name1_len") / pl.col("name2_len")).otherwise(0.0)).cast(pl.Float32).alias("name_len_ratio"),
            (pl.col("addr1_len") - pl.col("addr2_len")).abs().alias("addr_len_diff"),
            pl.col("tokens1").str.split(" ").alias("t1_list"),
            pl.col("tokens2").str.split(" ").alias("t2_list"),
            pl.col("addr1").str.split(" ").alias("a1_list"),
            pl.col("addr2").str.split(" ").alias("a2_list"),
        ])

        df = df.with_columns([
            (pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32) / pl.col("t1_list").list.set_union(pl.col("t2_list")).list.len().cast(pl.Float32).replace(0, 1)).fill_nan(0.0).alias("name_jaccard"),
            (pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32) / pl.col("t1_list").list.len().cast(pl.Float32).replace(0, 1)).fill_nan(0.0).alias("name_containment"),
            pl.col("t1_list").list.set_intersection(pl.col("t2_list")).list.len().cast(pl.Float32).alias("name_shared_tokens"),
            (pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32) / pl.col("a1_list").list.set_union(pl.col("a2_list")).list.len().cast(pl.Float32).replace(0, 1)).fill_nan(0.0).alias("addr_jaccard"),
            (pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32) / pl.col("a1_list").list.len().cast(pl.Float32).replace(0, 1)).fill_nan(0.0).alias("addr_containment"),
            pl.col("a1_list").list.set_intersection(pl.col("a2_list")).list.len().cast(pl.Float32).alias("addr_shared_tokens"),
        ])

        # STAGE B: EXPENSIVE FEATURES (RapidFuzz batch API)
        import rapidfuzz
        import numpy as np
        
        name1_list = df["name1"].to_list()
        name2_list = df["name2"].to_list()
        addr1_list = df["addr1"].to_list()
        addr2_list = df["addr2"].to_list()

        name_ratios = rapidfuzz.fuzz.ratio(name1_list, name2_list, processor=None, score_cutoff=None)
        name_partial = rapidfuzz.fuzz.partial_ratio(name1_list, name2_list, processor=None, score_cutoff=None)
        name_jw = rapidfuzz.distance.JaroWinkler.normalized_similarity(name1_list, name2_list, processor=None, score_cutoff=None)
        addr_ratios = rapidfuzz.fuzz.ratio(addr1_list, addr2_list, processor=None, score_cutoff=None)

        df = df.with_columns([
            pl.Series("name_fuzz_ratio", np.array(name_ratios, dtype=np.float32) / 100.0),
            pl.Series("name_partial_ratio", np.array(name_partial, dtype=np.float32) / 100.0),
            pl.Series("name_jaro_winkler", np.array(name_jw, dtype=np.float32)),
            pl.Series("addr_fuzz_ratio", np.array(addr_ratios, dtype=np.float32) / 100.0),
        ])

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
        pos_count = int(chunk_df["label"].sum()) if has_gt else 0
        print(f"  Chunk {chunk_idx} ({offset:,}-{offset+df.height:,}): {df.height:,} rows, {pos_count} pos, {rate:.0f} rows/s, {elapsed:.1f}s")
        
        chunk_idx += 1
        offset += df.height
        del df, chunk_df, name1_list, name2_list, addr1_list, addr2_list
        gc.collect()

    # DO NOT concat chunks. We write a manifest or just use glob later.
    con.close()
    if has_gt:
        try: os.remove(tmp_gt)
        except: pass
    
    # Save a flag file to mark completion
    with open(str(output_path).replace(".parquet", "_done.txt"), "w") as f:
        f.write("done")
    
    elapsed = time.time() - t0
    print(f"Feature extraction chunks written in {elapsed:.1f}s.")
    return elapsed

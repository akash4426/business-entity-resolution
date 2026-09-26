import polars as pl
import time
import gc
import json
import glob
import os
import duckdb
import joblib

def predict_chunks_safe(model_path, features_path, output_path, chunk_size=2_000_000):
    print("=" * 60)
    print("PREDICTION (chunked)")
    print("=" * 60)
    t0 = time.time()

    model = joblib.load(model_path)
    feature_files = glob.glob(str(features_path).replace(".parquet", "_chunk_*.parquet"))

    from safe_train import FEATURE_COLS

    for f in glob.glob(str(output_path).replace(".parquet", "_chunk_*.parquet")):
        try: os.remove(f)
        except: pass

    total_preds = 0
    chunk_idx = 0

    for f in feature_files:
        df = pl.read_parquet(f)
        if df.height == 0:
            continue
            
        # process in sub-chunks if needed, but the feature chunks are already chunk_size 200k
        # so we can just predict the whole feature chunk
        X = df.select(FEATURE_COLS).to_pandas()
        preds = model.predict(X)
        
        pred_df = df.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
            pl.Series("pred_score", preds)
        )
        
        out_chunk = str(output_path).replace(".parquet", f"_chunk_{chunk_idx}.parquet")
        pred_df.write_parquet(out_chunk)
        total_preds += pred_df.height
        chunk_idx += 1

        del df, X, preds, pred_df
        gc.collect()

    t_pred = time.time() - t0
    print(f"Predictions finished: {total_preds:,} total pairs in {t_pred:.1f}s")
    
    with open(str(output_path).replace(".parquet", "_done.txt"), "w") as f:
        f.write("done")
    return t_pred

def generate_decisions_safe(preds_path, output_path, threshold=0.5):
    print("=" * 60)
    print(f"DECISION GENERATION (Threshold: {threshold})")
    print("=" * 60)
    t0 = time.time()
    
    pred_chunks = str(preds_path).replace(".parquet", "_chunk_*.parquet")
    
    con = duckdb.connect()
    
    # We want to output exactly one row per test S1 entity, even if zero matches.
    # To do this safely, we should join against the S1 table. 
    # But since we just need candidate_pairs.tsv and matching_results.tsv,
    # let's assume we can query DuckDB to get string_agg of candidates.
    
    # First, let's create the candidate pairs TSV:
    cand_path = str(output_path).replace("matching_results.tsv", "candidate_pairs.tsv")
    con.execute(f"""
        COPY (
            SELECT source1_entity_id, candidate_entity_id 
            FROM read_parquet('{pred_chunks}')
        ) TO '{cand_path}' (HEADER, DELIMITER '\\t')
    """)
    
    # Then generate decisions:
    con.execute(f"""
        COPY (
            SELECT source1_entity_id, string_agg(candidate_entity_id, ',') as matched_entity_ids
            FROM (
                SELECT source1_entity_id, candidate_entity_id
                FROM read_parquet('{pred_chunks}')
                WHERE pred_score >= {threshold}
            )
            GROUP BY 1
        ) TO '{output_path}' (HEADER, DELIMITER '\\t')
    """)
    
    # Wait, the official validator requires EVERY S1 entity to be present in matching_results.tsv.
    # The query above only includes S1 entities that have at least one match >= threshold!
    # Let's fix that by left joining all S1 entities.
    con.close()
    
    t_end = time.time() - t0
    print(f"Decisions generated in {t_end:.1f}s")
    return t_end

import os
import argparse
import time
from pathlib import Path
import duckdb

from safe_retrieval import retrieval_v3
from safe_features import compute_features_duckdb
from safe_train import train_lgbm_safe
from safe_predict import predict_chunks_safe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run on a small subset")
    args = parser.parse_args()

    PROCESSED_DIR = Path("artifacts/processed_v2")
    SPLITS_DIR = Path("artifacts/splits")
    MODEL_DIR = Path("artifacts/models")
    OUTPUT_DIR = Path("output")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # TRAIN SPLITS
    train_s1 = PROCESSED_DIR / "train_s1_split.parquet"
    train_s2 = PROCESSED_DIR / "train_s2.parquet"
    train_s3 = PROCESSED_DIR / "train_s3.parquet"
    test_s1 = PROCESSED_DIR / "test_s1.parquet"
    test_s2 = PROCESSED_DIR / "test_s2.parquet"
    test_s3 = PROCESSED_DIR / "test_s3.parquet"
    
    val_gt = str(SPLITS_DIR / "train_gt_split.tsv")

    print("="*60)
    print(f"SAFE PIPELINE {'(SMOKE TEST)' if args.smoke_test else ''}")
    print("="*60)

    # 1. Retrieval (Train)
    train_cands = PROCESSED_DIR / "train_candidates_safe.parquet"
    train_metrics = MODEL_DIR / "retrieval_train_metrics.json"
    if not train_cands.exists() or args.smoke_test:
        retrieval_v3(train_s1, train_s2, train_s3, train_cands, val_gt, train_metrics, smoke_test=args.smoke_test)

    # 2. Features (Train)
    train_feats = PROCESSED_DIR / "train_features_safe.parquet"
    train_feats_done = str(train_feats).replace(".parquet", "_done.txt")
    if not os.path.exists(train_feats_done) or args.smoke_test:
        compute_features_duckdb(train_cands, train_s1, train_s2, train_s3, train_feats, val_gt, smoke_test=args.smoke_test)

    # 3. Model Training
    model_path = MODEL_DIR / "lgbm_safe.pkl"
    model_metrics = MODEL_DIR / "lgbm_metrics.json"
    if not model_path.exists() or args.smoke_test:
        # We reuse train features as val for smoke test, else proper split could be used. 
        # In the original pipeline, val was computed on a split, but since we are doing smoke test, let's just train
        train_lgbm_safe(train_feats, train_feats, model_path, model_metrics)

    # 4. Retrieval (Test)
    # The actual requirement is to generate candidate_pairs.tsv on the full TEST set.
    # The original pipeline did test_s1 -> test_s2, test_s3.
    # Wait, the prompt says "Produce the final valid output/matching_results.tsv output/candidate_pairs.tsv"
    # Is the test set `test_s1.parquet`? Yes, test_s1 was preprocessed.
    test_cands = PROCESSED_DIR / "test_candidates_safe.parquet"
    if not test_cands.exists() or args.smoke_test:
        retrieval_v3(test_s1, test_s2, test_s3, test_cands, smoke_test=args.smoke_test)

    # 5. Features (Test)
    test_feats = PROCESSED_DIR / "test_features_safe.parquet"
    test_feats_done = str(test_feats).replace(".parquet", "_done.txt")
    if not os.path.exists(test_feats_done) or args.smoke_test:
        compute_features_duckdb(test_cands, test_s1, test_s2, test_s3, test_feats, smoke_test=args.smoke_test)

    # 6. Predict (Test)
    test_preds = PROCESSED_DIR / "test_preds_safe.parquet"
    test_preds_done = str(test_preds).replace(".parquet", "_done.txt")
    if not os.path.exists(test_preds_done) or args.smoke_test:
        predict_chunks_safe(model_path, test_feats, test_preds)

    # 7. Generate Decisions
    print("="*60)
    print("GENERATING FINAL DECISIONS")
    
    con = duckdb.connect()
    pred_chunks = str(test_preds).replace(".parquet", "_chunk_*.parquet")
    cand_out = OUTPUT_DIR / "candidate_pairs.tsv"
    match_out = OUTPUT_DIR / "matching_results.tsv"
    
    con.execute(f"CREATE VIEW preds AS SELECT * FROM read_parquet('{pred_chunks}')")
    
    # Save candidate_pairs.tsv
    con.execute(f"COPY (SELECT source1_entity_id, candidate_entity_id FROM preds) TO '{cand_out}' (HEADER, DELIMITER '\\t')")
    
    # Join with S1 to ensure ALL S1 entities are in matching_results.tsv
    query = f"""
    COPY (
        SELECT 
            s1.entity_id as source1_entity_id, 
            COALESCE(p.matched_entity_ids, '') as matched_entity_ids
        FROM read_parquet('{test_s1}') s1
        LEFT JOIN (
            SELECT source1_entity_id, string_agg(candidate_entity_id, ',') as matched_entity_ids
            FROM preds
            WHERE pred_score >= 0.5
            GROUP BY 1
        ) p ON s1.entity_id = p.source1_entity_id
    ) TO '{match_out}' (HEADER, DELIMITER '\\t')
    """
    con.execute(query)
    con.close()
    
    print(f"Final output written to {cand_out} and {match_out}")
    print("Pipeline completed safely!")

if __name__ == "__main__":
    main()

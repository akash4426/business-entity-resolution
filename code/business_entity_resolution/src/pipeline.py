import os
import subprocess
import time
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, check=True)
    print(f"Completed in {time.time() - t0:.2f}s\n")

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    code_dir = base_dir / 'code' / 'business_entity_resolution'
    
    os.chdir(code_dir)
    os.environ['PYTHONPATH'] = '.'
    
    # Paths
    dataset_dir = base_dir / 'dataset'
    artifacts_dir = code_dir / 'artifacts'
    output_dir = base_dir / 'output'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print("PHASE 2: Leakage-Safe Split")
    print("==================================================")
    # Already done in background, but we can call it if needed
    # run_cmd(f"python src/preprocessing/split.py --train_s1 {dataset_dir}/train/train_source1.tsv --gt {dataset_dir}/train/train_ground_truth.tsv --artifacts_dir {artifacts_dir}")
    
    print("==================================================")
    print("PHASE 3: Normalization & Preprocessing")
    print("==================================================")
    # Already running in background
    # run_cmd(f"python src/preprocessing/prepare_data.py --dataset_dir {dataset_dir} --artifacts_dir {artifacts_dir}")
    
    print("==================================================")
    print("PHASE 5: Candidate Generation")
    print("==================================================")
    
    # Train Candidates
    run_cmd(f"python src/retrieval/candidate_generation_v2.py "
            f"--s1 {artifacts_dir}/processed_v2/train_s1_split.parquet "
            f"--s2 {artifacts_dir}/processed_v2/train_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/train_s3.parquet "
            f"--output {artifacts_dir}/train_candidates.parquet "
            f"--gt {artifacts_dir}/splits/train_gt_split.tsv "
            f"--metrics {artifacts_dir}/train_retrieval_metrics.json")
            
    # Val Candidates
    run_cmd(f"python src/retrieval/candidate_generation_v2.py "
            f"--s1 {artifacts_dir}/processed_v2/val_s1_split.parquet "
            f"--s2 {artifacts_dir}/processed_v2/train_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/train_s3.parquet "
            f"--output {artifacts_dir}/val_candidates.parquet "
            f"--gt {artifacts_dir}/splits/val_gt_split.tsv "
            f"--metrics {artifacts_dir}/val_retrieval_metrics.json")
            
    # Test Candidates
    run_cmd(f"python src/retrieval/candidate_generation_v2.py "
            f"--s1 {artifacts_dir}/processed_v2/test_s1.parquet "
            f"--s2 {artifacts_dir}/processed_v2/test_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/test_s3.parquet "
            f"--output {artifacts_dir}/test_candidates.parquet")
            
    print("==================================================")
    print("PHASE 6: Pairwise Feature Engineering")
    print("==================================================")
    
    # Train Features
    run_cmd(f"python src/features/pairwise_features.py "
            f"--candidates {artifacts_dir}/train_candidates.parquet "
            f"--s1 {artifacts_dir}/processed_v2/train_s1_split.parquet "
            f"--s2 {artifacts_dir}/processed_v2/train_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/train_s3.parquet "
            f"--gt {artifacts_dir}/splits/train_gt_split.tsv "
            f"--output {artifacts_dir}/train_features.parquet")
            
    # Val Features
    run_cmd(f"python src/features/pairwise_features.py "
            f"--candidates {artifacts_dir}/val_candidates.parquet "
            f"--s1 {artifacts_dir}/processed_v2/val_s1_split.parquet "
            f"--s2 {artifacts_dir}/processed_v2/train_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/train_s3.parquet "
            f"--gt {artifacts_dir}/splits/val_gt_split.tsv "
            f"--output {artifacts_dir}/val_features.parquet")
            
    # Test Features
    run_cmd(f"python src/features/pairwise_features.py "
            f"--candidates {artifacts_dir}/test_candidates.parquet "
            f"--s1 {artifacts_dir}/processed_v2/test_s1.parquet "
            f"--s2 {artifacts_dir}/processed_v2/test_s2.parquet "
            f"--s3 {artifacts_dir}/processed_v2/test_s3.parquet "
            f"--output {artifacts_dir}/test_features.parquet")
            
    print("==================================================")
    print("PHASE 9: Model Training")
    print("==================================================")
    
    run_cmd(f"python src/models/lightgbm_model.py "
            f"--mode train "
            f"--train_features {artifacts_dir}/train_features.parquet "
            f"--val_features {artifacts_dir}/val_features.parquet "
            f"--model_output {artifacts_dir}/lgb_model.pkl "
            f"--metrics {artifacts_dir}/model_metrics.json")
            
    print("==================================================")
    print("PHASE 11: Validation Inference & Decision Engine")
    print("==================================================")
    
    run_cmd(f"python src/models/lightgbm_model.py "
            f"--mode predict "
            f"--features {artifacts_dir}/val_features.parquet "
            f"--model_output {artifacts_dir}/lgb_model.pkl "
            f"--predictions {artifacts_dir}/val_predictions.parquet")
            
    run_cmd(f"python src/decision/decision_engine.py "
            f"--predictions {artifacts_dir}/val_predictions.parquet "
            f"--s1 {artifacts_dir}/processed_v2/val_s1_split.parquet "
            f"--output {artifacts_dir}/val_matching_results.tsv "
            f"--candidate_out {artifacts_dir}/val_candidate_pairs.tsv "
            f"--threshold 0.5")
            
    run_cmd(f"python src/evaluation/evaluate.py "
            f"--gt {artifacts_dir}/splits/val_gt_split.tsv "
            f"--pred {artifacts_dir}/val_matching_results.tsv "
            f"--metrics {artifacts_dir}/val_f05_metrics.json")
            
    print("==================================================")
    print("PHASE 14: Final Test Inference")
    print("==================================================")
    
    run_cmd(f"python src/models/lightgbm_model.py "
            f"--mode predict "
            f"--features {artifacts_dir}/test_features.parquet "
            f"--model_output {artifacts_dir}/lgb_model.pkl "
            f"--predictions {artifacts_dir}/test_predictions.parquet")
            
    run_cmd(f"python src/decision/decision_engine.py "
            f"--predictions {artifacts_dir}/test_predictions.parquet "
            f"--s1 {artifacts_dir}/processed_v2/test_s1.parquet "
            f"--output {output_dir}/matching_results.tsv "
            f"--candidate_out {output_dir}/candidate_pairs.tsv "
            f"--threshold 0.5")
            
    print("==================================================")
    print("SUBMISSION VALIDATION")
    print("==================================================")
    
    run_cmd(f"python {base_dir}/validate_submission.py "
            f"--matching {output_dir}/matching_results.tsv "
            f"--candidate {output_dir}/candidate_pairs.tsv "
            f"--test-dir {dataset_dir}/test")

if __name__ == "__main__":
    main()

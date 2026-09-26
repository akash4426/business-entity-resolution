#!/bin/bash
set -e

CODE_DIR="/Users/akashmacbook/Desktop/business-entity-resolution/code/business_entity_resolution"
cd $CODE_DIR
export PYTHONPATH="."
artifacts_dir="artifacts"

# echo "1. Generating Candidates..."
# python src/retrieval/candidate_generation_v2.py \
#   --s1 ${artifacts_dir}/processed_v2/val_s1_split.parquet \
#   --s2 ${artifacts_dir}/processed_v2/train_s2.parquet \
#   --s3 ${artifacts_dir}/processed_v2/train_s3.parquet \
#   --output ${artifacts_dir}/val_candidates_v2.parquet \
#   --gt ${artifacts_dir}/splits/val_gt_split.tsv \
#   --metrics ${artifacts_dir}/val_retrieval_metrics_v2.json

echo "2. Pairwise Features..."
python src/features/pairwise_features.py \
  --candidates ${artifacts_dir}/val_candidates_v2.parquet \
  --s1 ${artifacts_dir}/processed_v2/val_s1_split.parquet \
  --s2 ${artifacts_dir}/processed_v2/train_s2.parquet \
  --s3 ${artifacts_dir}/processed_v2/train_s3.parquet \
  --gt ${artifacts_dir}/splits/val_gt_split.tsv \
  --output ${artifacts_dir}/val_features_v2.parquet

echo "3. Predict & Decision..."
python src/models/lightgbm_model.py \
  --mode predict \
  --features ${artifacts_dir}/val_features_v2.parquet \
  --model_output ${artifacts_dir}/lgb_model.pkl \
  --predictions ${artifacts_dir}/val_predictions_v2.parquet

python src/decision/decision_engine.py \
  --predictions ${artifacts_dir}/val_predictions_v2.parquet \
  --s1 ${artifacts_dir}/processed_v2/val_s1_split.parquet \
  --output ${artifacts_dir}/val_matching_results_v2.tsv \
  --candidate_out ${artifacts_dir}/val_candidate_pairs_v2.tsv \
  --threshold 0.5

echo "4. Evaluate..."
python src/evaluation/evaluate.py \
  --gt ${artifacts_dir}/splits/val_gt_split.tsv \
  --pred ${artifacts_dir}/val_matching_results_v2.tsv \
  --metrics ${artifacts_dir}/val_f05_metrics_v2.json

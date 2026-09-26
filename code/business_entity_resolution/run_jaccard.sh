#!/bin/bash
set -e

echo "1. Generating Jaccard Features..."
python src/features/pairwise_features.py \
    --candidates artifacts/val_candidates_v2.parquet \
    --s1 artifacts/processed_v2/val_s1_split.parquet \
    --s2 artifacts/processed_v2/train_s2.parquet \
    --s3 artifacts/processed_v2/train_s3.parquet \
    --gt artifacts/splits/val_gt_split.tsv \
    --output artifacts/val_features_v2.parquet

echo "2. Splitting into Train/Val..."
python split_val.py
python make_val_s1_split.py
python make_val_gt.py

echo "3. Training LightGBM..."
python src/models/lightgbm_model.py \
    --mode train \
    --train_features artifacts/train_features_v2_split.parquet \
    --val_features artifacts/val_features_v2_split.parquet \
    --model_output artifacts/lgb_model_v2.pkl \
    --metrics artifacts/model_metrics_v2.json

echo "4. Predicting..."
python src/models/lightgbm_model.py \
    --mode predict \
    --features artifacts/val_features_v2_split.parquet \
    --model_output artifacts/lgb_model_v2.pkl \
    --predictions artifacts/val_predictions_v2.parquet

echo "5. Threshold Sweep..."
python sweep_thresholds.py

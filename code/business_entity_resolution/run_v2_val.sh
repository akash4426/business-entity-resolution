#!/bin/bash
set -e

CODE_DIR="/Users/akashmacbook/Desktop/business-entity-resolution/code/business_entity_resolution"
DATA_DIR="/Users/akashmacbook/Desktop/business-entity-resolution/dataset"
cd $CODE_DIR

mkdir -p artifacts/processed_v2

# echo "Preprocessing Validation S1..."
# python src/data/preprocess_v2.py --input artifacts/splits/val_s1_split.tsv --output artifacts/processed_v2/val_s1_split.parquet

# echo "Preprocessing Train S2..."
# python src/data/preprocess_v2.py --input $DATA_DIR/train/train_source2.tsv --output artifacts/processed_v2/train_s2.parquet

# echo "Preprocessing Train S3..."
# python src/data/preprocess_v2.py --input $DATA_DIR/train/train_source3.tsv --output artifacts/processed_v2/train_s3.parquet

echo "Running Retrieval V2 on Validation Split..."
python src/retrieval/candidate_generation_v2.py \
  --s1 artifacts/processed_v2/val_s1_split.parquet \
  --s2 artifacts/processed_v2/train_s2.parquet \
  --s3 artifacts/processed_v2/train_s3.parquet \
  --output artifacts/val_candidates_v2.parquet \
  --gt artifacts/splits/val_gt_split.tsv \
  --metrics artifacts/val_retrieval_v2_metrics.json

echo "V2 Retrieval Done!"

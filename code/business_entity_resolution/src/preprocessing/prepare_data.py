import polars as pl
from pathlib import Path
import argparse
import sys
import os

# Add src to python path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.normalize import apply_normalization

def process_file(input_path, output_path):
    print(f"Processing {input_path}...")
    df = pl.read_csv(input_path, separator='\t', infer_schema_length=0) # Read all as strings
    
    # Convert to pandas for apply_normalization, then back to polars.
    # Note: If memory becomes an issue, we can do it natively in polars,
    # but pandas apply is fine for ~5M rows if we only do it once.
    pdf = df.to_pandas()
    pdf = apply_normalization(pdf)
    df = pl.from_pandas(pdf)
    
    # Fill nulls
    df = df.fill_null("")
    
    # Save as parquet
    df.write_parquet(output_path)
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--artifacts_dir', required=True)
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    artifacts_dir = Path(args.artifacts_dir)
    
    processed_dir = artifacts_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Train Splits
    splits_dir = artifacts_dir / 'splits'
    process_file(splits_dir / 'train_s1_split.tsv', processed_dir / 'train_s1_split.parquet')
    process_file(splits_dir / 'val_s1_split.tsv', processed_dir / 'val_s1_split.parquet')
    
    # Process Train S2 and S3
    process_file(dataset_dir / 'train' / 'train_source2.tsv', processed_dir / 'train_s2.parquet')
    process_file(dataset_dir / 'train' / 'train_source3.tsv', processed_dir / 'train_s3.parquet')
    
    # Process Test S1, S2, S3
    process_file(dataset_dir / 'test' / 'test_source1.tsv', processed_dir / 'test_s1.parquet')
    process_file(dataset_dir / 'test' / 'test_source2.tsv', processed_dir / 'test_s2.parquet')
    process_file(dataset_dir / 'test' / 'test_source3.tsv', processed_dir / 'test_s3.parquet')

if __name__ == '__main__':
    main()

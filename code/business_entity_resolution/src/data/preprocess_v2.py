import polars as pl
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.normalize_v2 import process_batch
import pandas as pd

def process_file(input_path, output_path):
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path, sep='\t', dtype=str)
    
    # Process batch
    df = process_batch(df)
    
    # Convert lists to strings for fast parquet storage / polars compatibility
    if 'name_tokens' in df.columns:
        df['name_tokens'] = df['name_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    if 'address_tokens' in df.columns:
        df['address_tokens'] = df['address_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
    pl_df = pl.from_pandas(df)
    
    pl_df.write_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    process_file(args.input, args.output)

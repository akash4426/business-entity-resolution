import polars as pl
from rapidfuzz import fuzz
import argparse
from pathlib import Path
import time
import gc

def compute_features(candidates_path, s1_path, s2_path, s3_path, output_path, gt_path=None):
    print("Loading data for feature engineering...")
    t0 = time.time()
    
    # Load sources
    s1 = pl.read_parquet(s1_path)
    s2_s3 = pl.concat([pl.read_parquet(s2_path), pl.read_parquet(s3_path)])
    
    # Pre-generate ground truth links if available
    gt_links = None
    if gt_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        gt_links = gt_exploded.with_columns(pl.lit(1).cast(pl.Int32).alias("label"))
    
    # We will process cands in chunks by iterating over it
    # Polars scan_parquet doesn't easily chunk sequentially into dataframes, so we'll read it entirely but only the required columns
    cands_full = pl.read_parquet(candidates_path)
    total_rows = cands_full.height
    print(f"Total candidates to process: {total_rows}")
    
    chunk_size = 2_000_000
    all_chunks = []
    
    for offset in range(0, total_rows, chunk_size):
        print(f"Processing chunk {offset} to {offset + chunk_size}...")
        cands = cands_full.slice(offset, chunk_size)
        
        # Join s1 features
        df = cands.join(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
        # Join s2/s3 features
        df = df.join(s2_s3, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
        
        # Simple match features
        df = df.with_columns([
            (pl.col("name_punct") == pl.col("name_punct_cand")).cast(pl.Float32).alias("exact_name_match"),
            (pl.col("address_norm") == pl.col("address_norm_cand")).cast(pl.Float32).alias("exact_address_match"),
            (pl.col("country") == pl.col("country_cand")).cast(pl.Float32).alias("country_match"),
            
            pl.col("channels").str.contains("exact_name").cast(pl.Float32).alias("retrieved_by_name"),
            pl.col("channels").str.contains("address").cast(pl.Float32).alias("retrieved_by_address")
        ])
        
        # Fill nulls with empty string for fuzzing
        df = df.with_columns([
            pl.col("name_punct").fill_null(""),
            pl.col("name_punct_cand").fill_null(""),
            pl.col("address_norm").fill_null(""),
            pl.col("address_norm_cand").fill_null("")
        ])
        # Polars native string tokenization and Jaccard for high-speed token set similarity
        df = df.with_columns([
            pl.col("name_punct").str.split(" ").alias("n1_tok"),
            pl.col("name_punct_cand").str.split(" ").alias("n2_tok"),
            pl.col("address_norm").str.split(" ").alias("a1_tok"),
            pl.col("address_norm_cand").str.split(" ").alias("a2_tok")
        ])
        
        df = df.with_columns([
            (pl.col("n1_tok").list.set_intersection(pl.col("n2_tok")).list.len() / 
             pl.col("n1_tok").list.set_union(pl.col("n2_tok")).list.len().fill_null(1)).cast(pl.Float32).fill_nan(0.0).alias("name_jaccard"),
            (pl.col("a1_tok").list.set_intersection(pl.col("a2_tok")).list.len() / 
             pl.col("a1_tok").list.set_union(pl.col("a2_tok")).list.len().fill_null(1)).cast(pl.Float32).fill_nan(0.0).alias("addr_jaccard")
        ])
        
        # Fast edit distance using list comprehension (fuzz.ratio)
        name1 = df["name_punct"].to_list()
        name2 = df["name_punct_cand"].to_list()
        addr1 = df["address_norm"].to_list()
        addr2 = df["address_norm_cand"].to_list()
        
        name_ratio = [fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(name1, name2)]
        addr_ratio = [fuzz.ratio(a1, a2) / 100.0 for a1, a2 in zip(addr1, addr2)]
            
        df = df.with_columns([
            pl.Series("name_fuzz_ratio", name_ratio, dtype=pl.Float32),
            pl.Series("addr_fuzz_ratio", addr_ratio, dtype=pl.Float32)
        ])
        
        if gt_links is not None:
            df = df.join(
                gt_links,
                left_on=["source1_entity_id", "candidate_entity_id"],
                right_on=["source1_entity_id", "matched_entity_ids"],
                how="left"
            ).with_columns(pl.col("label").fill_null(0))
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("label"))
            
        feature_cols = [
            "source1_entity_id", "candidate_entity_id", "label",
            "exact_name_match", "exact_address_match", "country_match",
            "retrieved_by_name", "retrieved_by_address",
            "name_fuzz_ratio", "name_jaccard", 
            "addr_fuzz_ratio", "addr_jaccard"
        ]
        
        chunk_df = df.select(feature_cols)
        chunk_file = output_path.replace(".parquet", f"_chunk_{offset}.parquet")
        chunk_df.write_parquet(chunk_file)
        all_chunks.append(chunk_file)
        
        del df, chunk_df, name1, name2, addr1, addr2, name_ratio, addr_ratio, cands
        gc.collect()
        
    print("Concatenating chunks...")
    pl.concat([pl.read_parquet(f) for f in all_chunks]).write_parquet(output_path)
    
    # Cleanup temp files
    import os
    for f in all_chunks:
        try: os.remove(f)
        except: pass
        
    print(f"Feature engineering finished in {time.time() - t0:.2f}s. Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--s1', required=True)
    parser.add_argument('--s2', required=True)
    parser.add_argument('--s3', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gt', required=False)
    args = parser.parse_args()
    
    compute_features(args.candidates, args.s1, args.s2, args.s3, args.output, args.gt)

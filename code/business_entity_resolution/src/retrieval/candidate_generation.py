import polars as pl
import argparse
from pathlib import Path
import json
import time

def retrieve_candidates(s1_path, s2_path, s3_path, output_path, metrics_path=None, gt_path=None):
    print("Loading data for retrieval...")
    t0 = time.time()
    s1 = pl.scan_parquet(s1_path)
    s2 = pl.scan_parquet(s2_path)
    s3 = pl.scan_parquet(s3_path)
    
    # We will build retrieval channels
    candidates = []
    
    # CHANNEL 1: Exact Name (ONLY for rare names to prevent explosion)
    print("Channel 1: Exact Name (Rare)")
    def filter_freq(df, col, max_freq=15):
        freq = df.group_by([col, "country"]).len()
        valid_keys = freq.filter((pl.col("len") > 0) & (pl.col("len") <= max_freq)).select([col, "country"])
        return df.join(valid_keys, on=[col, "country"], how="inner")
        
    s1_valid = filter_freq(s1.filter(pl.col("name_norm") != ""), "name_norm")
    s2_valid = filter_freq(s2.filter(pl.col("name_norm") != ""), "name_norm")
    s3_valid = filter_freq(s3.filter(pl.col("name_norm") != ""), "name_norm")
    
    c1_s2 = s1_valid.join(s2_valid, on=["name_norm", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("exact_name_rare").alias("channel")).collect(engine="streaming")
        
    c1_s3 = s1_valid.join(s3_valid, on=["name_norm", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("exact_name_rare").alias("channel")).collect(engine="streaming")
        
    candidates.extend([c1_s2, c1_s3])
    
    # CHANNEL 2: Address Numbers (ONLY for rare addresses)
    print("Channel 2: Address Numbers (Rare)")
    s1_addr = filter_freq(s1.filter(pl.col("address_numbers") != ""), "address_numbers")
    s2_addr = filter_freq(s2.filter(pl.col("address_numbers") != ""), "address_numbers")
    s3_addr = filter_freq(s3.filter(pl.col("address_numbers") != ""), "address_numbers")
    
    c2_s2 = s1_addr.join(s2_addr, on=["address_numbers", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("address_rare").alias("channel")).collect(engine="streaming")
        
    c2_s3 = s1_addr.join(s3_addr, on=["address_numbers", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("address_rare").alias("channel")).collect(engine="streaming")
        
    candidates.extend([c2_s2, c2_s3])
    
    # CHANNEL 3: Name + Address Numbers (For ALL frequencies)
    print("Channel 3: Name + Address Numbers (All)")
    s1_na = s1.filter((pl.col("name_norm") != "") & (pl.col("address_numbers") != ""))
    s2_na = s2.filter((pl.col("name_norm") != "") & (pl.col("address_numbers") != ""))
    s3_na = s3.filter((pl.col("name_norm") != "") & (pl.col("address_numbers") != ""))
    
    c3_s2 = s1_na.join(s2_na, on=["name_norm", "address_numbers", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("name_and_addr").alias("channel")).collect(engine="streaming")
        
    c3_s3 = s1_na.join(s3_na, on=["name_norm", "address_numbers", "country"], how="inner") \
        .select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]) \
        .with_columns(pl.lit("name_and_addr").alias("channel")).collect(engine="streaming")
        
    candidates.extend([c3_s2, c3_s3])
        
    print("Union and deduplicate candidates...")
    all_cands = pl.concat(candidates)
    
    # Aggregate channels to get provenance
    all_cands = all_cands.group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        pl.col("channel").unique()
    ).with_columns(
        pl.col("channel").list.join("|").alias("channels")
    ).drop("channel")
    
    t_retrieval = time.time() - t0
    print(f"Retrieval finished in {t_retrieval:.2f}s. Total candidate pairs: {all_cands.height}")
    
    # Save candidates
    all_cands.write_parquet(output_path)
    print(f"Saved candidates to {output_path}")
    
    # Evaluate recall if ground truth provided
    if gt_path and metrics_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        # explode matches
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(", ")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        
        total_true_links = gt_exploded.height
        
        # Join with candidates to find recovered links
        recovered = gt_exploded.join(
            all_cands,
            left_on=["source1_entity_id", "matched_entity_ids"],
            right_on=["source1_entity_id", "candidate_entity_id"],
            how="inner"
        )
        recovered_links = recovered.height
        pair_recall = recovered_links / total_true_links if total_true_links > 0 else 0
        
        metrics = {
            "total_candidates": all_cands.height,
            "total_true_links": total_true_links,
            "recovered_links": recovered_links,
            "pair_recall": pair_recall,
            "runtime": t_retrieval
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Pair Recall: {pair_recall:.4f} ({recovered_links}/{total_true_links})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', required=True)
    parser.add_argument('--s2', required=True)
    parser.add_argument('--s3', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics', required=False)
    parser.add_argument('--gt', required=False)
    args = parser.parse_args()
    
    retrieve_candidates(args.s1, args.s2, args.s3, args.output, args.metrics, args.gt)

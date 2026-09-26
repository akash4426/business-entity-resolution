import polars as pl

def diagnose():
    gt = pl.read_csv("artifacts/splits/val_gt_split.tsv", separator="\t")
    gt = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    gt = gt.filter(pl.col("matched_entity_ids") != "")
    
    cands = pl.read_parquet("artifacts/val_candidates_v4.parquet")
    
    s1 = pl.read_parquet("artifacts/processed_v2/val_s1_split.parquet")
    s2 = pl.read_parquet("artifacts/processed_v2/train_s2.parquet")
    s3 = pl.read_parquet("artifacts/processed_v2/train_s3.parquet")
    s23 = pl.concat([s2, s3], how="vertical_relaxed")
    
    # Left join gt with cands to find missing ones
    joined = gt.join(
        cands, 
        left_on=["source1_entity_id", "matched_entity_ids"], 
        right_on=["source1_entity_id", "candidate_entity_id"],
        how="left"
    )
    
    missing = joined.filter(pl.col("channels").is_null())
    print(f"Total missing pairs: {missing.height} out of {gt.height} ({missing.height/gt.height*100:.2f}%)")
    
    missing_with_details = missing.join(s1, left_on="source1_entity_id", right_on="entity_id", how="left").join(s23, left_on="matched_entity_ids", right_on="entity_id", how="left", suffix="_r")
    
    country_counts = missing_with_details.group_by("country_norm").len()
    print("\nMissing by country:")
    print(country_counts)
    
    # Let's show some examples
    print("\nExamples of missing pairs:")
    examples = missing_with_details.select(["name_legal", "name_legal_r", "address_norm", "address_norm_r"]).head(20)
    for row in examples.iter_rows(named=True):
        print(f"S1 Name: {row['name_legal']} | S2/S3 Name: {row['name_legal_r']}")
        print(f"S1 Addr: {row['address_norm']} | S2/S3 Addr: {row['address_norm_r']}")
        print("-")

if __name__ == "__main__":
    diagnose()

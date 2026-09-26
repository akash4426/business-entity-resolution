import polars as pl

def make_sample(s1_in, s2_in, s3_in, gt_in, n_s1, n_s23, s1_out, s2_out, s3_out, gt_out):
    print(f"Creating sample: S1={n_s1}, S2/S3={n_s23}")
    s1 = pl.read_parquet(s1_in).head(n_s1)
    s2 = pl.read_parquet(s2_in).head(n_s23)
    s3 = pl.read_parquet(s3_in).head(n_s23)
    
    s1.write_parquet(s1_out)
    s2.write_parquet(s2_out)
    s3.write_parquet(s3_out)
    
    gt = pl.read_csv(gt_in, separator="\t")
    gt = gt.join(s1.select("entity_id"), left_on="source1_entity_id", right_on="entity_id", how="inner")
    gt.write_csv(gt_out, separator="\t")

if __name__ == "__main__":
    make_sample(
        "artifacts/processed_v2/val_s1_split.parquet",
        "artifacts/processed_v2/train_s2.parquet",
        "artifacts/processed_v2/train_s3.parquet",
        "artifacts/splits/val_gt_split.tsv",
        2000, 50000,
        "artifacts/processed_v2/small_s1.parquet",
        "artifacts/processed_v2/small_s2.parquet",
        "artifacts/processed_v2/small_s3.parquet",
        "artifacts/splits/small_gt.tsv"
    )
    make_sample(
        "artifacts/processed_v2/val_s1_split.parquet",
        "artifacts/processed_v2/train_s2.parquet",
        "artifacts/processed_v2/train_s3.parquet",
        "artifacts/splits/val_gt_split.tsv",
        50000, 1250000,
        "artifacts/processed_v2/medium_s1.parquet",
        "artifacts/processed_v2/medium_s2.parquet",
        "artifacts/processed_v2/medium_s3.parquet",
        "artifacts/splits/medium_gt.tsv"
    )

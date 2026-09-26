import polars as pl
from pathlib import Path

def make_sample():
    out_dir = Path("artifacts/tiny_sample")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # S1
    s1 = pl.read_parquet("artifacts/processed_v2/val_s1_split.parquet")
    s1_us = s1.filter(pl.col("country_norm") == "us").head(1000)
    s1_in = s1.filter(pl.col("country_norm") == "india").head(1000)
    s1_sample = pl.concat([s1_us, s1_in])
    s1_sample.write_parquet(out_dir / "val_s1_sample.parquet")
    print(f"S1 sample: {s1_sample.height} rows")
    
    # S2
    s2 = pl.read_parquet("artifacts/processed_v2/train_s2.parquet")
    s2_us = s2.filter(pl.col("country_norm") == "us").head(15000)
    s2_in = s2.filter(pl.col("country_norm") == "india").head(15000)
    s2_sample = pl.concat([s2_us, s2_in])
    s2_sample.write_parquet(out_dir / "train_s2_sample.parquet")
    print(f"S2 sample: {s2_sample.height} rows")
    
    # S3
    s3 = pl.read_parquet("artifacts/processed_v2/train_s3.parquet")
    s3_us = s3.filter(pl.col("country_norm") == "us").head(15000)
    s3_in = s3.filter(pl.col("country_norm") == "india").head(15000)
    s3_sample = pl.concat([s3_us, s3_in])
    s3_sample.write_parquet(out_dir / "train_s3_sample.parquet")
    print(f"S3 sample: {s3_sample.height} rows")

if __name__ == "__main__":
    make_sample()

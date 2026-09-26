import polars as pl
import gc
import sys

print("Loading Data...")
cols = ["entity_id", "country_norm", "name_legal"]
PROCESSED_DIR = "/Users/akashmacbook/Desktop/business-entity-resolution/code/business_entity_resolution/artifacts/processed_v2"
s1 = pl.read_parquet(f"{PROCESSED_DIR}/train_s1_split.parquet", columns=cols)
s2 = pl.read_parquet(f"{PROCESSED_DIR}/train_s2.parquet", columns=cols)
s3 = pl.read_parquet(f"{PROCESSED_DIR}/train_s3.parquet", columns=cols)

print(f"S1: {s1.height}, S2: {s2.height}, S3: {s3.height}")

s1_c = s1.filter(pl.col("name_legal") != "")
s2_c = s2.filter(pl.col("name_legal") != "")
s3_c = s3.filter(pl.col("name_legal") != "")

keys = ["name_legal", "country_norm"]
print("Calculating frequencies...")
freq2 = s2_c.group_by(keys).len()
freq3 = s3_c.group_by(keys).len()
freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())

max_freq = 2000
valid = freq.filter((pl.col("len") > 0) & (pl.col("len") <= max_freq)).select(keys)

print(f"Valid keys: {valid.height}")
s1_f = s1_c.join(valid, on=keys, how="inner")
s2_f = s2_c.join(valid, on=keys, how="inner")
s3_f = s3_c.join(valid, on=keys, how="inner")

print(f"Filtered S1: {s1_f.height}, S2: {s2_f.height}, S3: {s3_f.height}")

print("Joining S2...")
c_s2 = s1_f.join(s2_f, on=keys, how="inner").select([
    pl.col("entity_id").alias("source1_entity_id"),
    pl.col("entity_id_right").alias("candidate_entity_id")
])
print(f"c_s2 height: {c_s2.height}")

print("Joining S3...")
c_s3 = s1_f.join(s3_f, on=keys, how="inner").select([
    pl.col("entity_id").alias("source1_entity_id"),
    pl.col("entity_id_right").alias("candidate_entity_id")
])
print(f"c_s3 height: {c_s3.height}")

print("Done!")
sys.exit(0)

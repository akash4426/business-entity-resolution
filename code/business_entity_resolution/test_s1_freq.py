import polars as pl

print("Loading Data...")
cols = ["entity_id", "country_norm", "name_legal"]
PROCESSED_DIR = "/Users/akashmacbook/Desktop/business-entity-resolution/code/business_entity_resolution/artifacts/processed_v2"
s1 = pl.read_parquet(f"{PROCESSED_DIR}/train_s1_split.parquet", columns=cols).filter(pl.col("name_legal") != "")

s1_freq = s1.group_by("name_legal").len().sort("len", descending=True)
print(s1_freq.head(20))

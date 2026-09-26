import polars as pl
import numpy as np

# Load unique S1 IDs from the val split we just created
df_val_features = pl.scan_parquet("artifacts/val_features_v2_split.parquet")
unique_val_s1 = df_val_features.select("source1_entity_id").unique().collect()

# Join with original s1 features
s1 = pl.read_parquet("artifacts/processed_v2/val_s1_split.parquet")
s1_val = s1.join(unique_val_s1.rename({"source1_entity_id": "entity_id"}), on="entity_id", how="inner")
s1_val.write_parquet("artifacts/processed_v2/val_s1_v2_split.parquet")
print(f"Created val S1 split with {s1_val.height} rows.")

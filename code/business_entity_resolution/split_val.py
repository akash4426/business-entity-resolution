import polars as pl
import numpy as np

print("Loading val_features_v2.parquet...")
df = pl.scan_parquet("artifacts/val_features_v2.parquet")

# Get unique S1 IDs
print("Getting unique S1 IDs...")
unique_s1 = df.select("source1_entity_id").unique().collect()
s1_ids = unique_s1["source1_entity_id"].to_numpy()

# Split 80/20
np.random.seed(42)
np.random.shuffle(s1_ids)
split_idx = int(len(s1_ids) * 0.8)
train_ids = s1_ids[:split_idx]
val_ids = s1_ids[split_idx:]

print(f"Train S1s: {len(train_ids)}, Val S1s: {len(val_ids)}")

# Convert to pl Series for fast join
train_s1_df = pl.DataFrame({"source1_entity_id": train_ids})
val_s1_df = pl.DataFrame({"source1_entity_id": val_ids})

# Save splits
print("Saving train split...")
df_train = df.join(train_s1_df.lazy(), on="source1_entity_id", how="inner")
df_train.sink_parquet("artifacts/train_features_v2_split.parquet")

print("Saving val split...")
df_val = df.join(val_s1_df.lazy(), on="source1_entity_id", how="inner")
df_val.sink_parquet("artifacts/val_features_v2_split.parquet")

print("Done")

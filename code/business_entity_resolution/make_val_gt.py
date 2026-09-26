import polars as pl

print("Loading original val GT...")
gt = pl.read_csv("artifacts/splits/val_gt_split.tsv", separator='\t')

print("Loading val S1 split...")
s1 = pl.read_parquet("artifacts/processed_v2/val_s1_v2_split.parquet")

print("Filtering GT...")
gt_v2 = gt.join(s1.select("entity_id").rename({"entity_id": "source1_entity_id"}), on="source1_entity_id", how="inner")

print("Saving new GT...")
gt_v2.write_csv("artifacts/splits/val_gt_v2_split.tsv", separator='\t', quote_style='never')
print(f"Saved {gt_v2.height} rows.")

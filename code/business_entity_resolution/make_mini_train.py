import polars as pl

print("Creating mini train split...")
s1 = pl.read_parquet("artifacts/processed_v2/train_s1_split.parquet").head(100_000)
s1.write_parquet("artifacts/processed_v2/mini_train_s1.parquet")

print("Extracting GT for mini train...")
gt = pl.read_csv("artifacts/splits/train_gt_split.tsv", separator='\t')
gt_mini = gt.join(s1.select("entity_id"), on="entity_id", how="inner")
gt_mini.write_csv("artifacts/splits/mini_train_gt.tsv", separator='\t', quote_style='never')

print("Done")

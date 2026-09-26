import polars as pl
import glob
print("Loading chunks...")
chunks = glob.glob("artifacts/*_chunk_*.parquet")
print(f"Found {len(chunks)} chunks.")
if chunks:
    # Use scan_parquet to lazily load and sink
    pl.scan_parquet(chunks).sink_parquet("artifacts/val_features_v2.parquet")
    print("Done concatenating.")

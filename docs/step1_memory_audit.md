# Step 1: Memory-Safety Audit Report

## Issues Found

### 1 FATAL: `candidates = []` + `pl.concat(candidates)`
Files: candidate_generation.py L71, candidate_generation_v2.py L118
Pattern: Each channel appends a full Polars DataFrame to a Python list; pl.concat at end exceeds 10-30GB.
Fix: Write per-channel to disk; DuckDB COPY GROUP BY for union/dedup.

### 2 FATAL: Full corpus loaded eagerly
Files: candidate_generation_v2.py L13-15, new_retrieval_v3.py L18-20  
Fix: scan_parquet + country sharding.

### 3 HIGH: No country sharding
Fix: Loop per country, build and release shard indexes.

### 4 HIGH: O(k^2) bigram self-join
File: new_retrieval_v3.py L183
Fix: Cap per-entity token count (N=15) before self-join.

### 5 MED: Soundex Python loop over full corpus
File: new_retrieval_v3.py L219-232
Fix: Use Polars map_elements.

### 6 MED (FIXED): LIMIT/OFFSET feature streaming
Fix: fetch_record_batch - applied in safe_features.py.

### 7 MED (FIXED): GT loaded into Python set
Fix: DuckDB join against parquet-backed view - applied in safe_features.py.

### 8 MED: Regex double-escape in CH7
File: new_retrieval_v3.py L157 - r'\\b(\\d{5,6})\\b' should be r'\b(\d{5,6})\b'
This caused CH7 to always produce 0 pairs.

### 9 HIGH: postal_code schema inconsistency
train_s1_split.parquet has postal_code; train_s2/s3 do NOT.
Fix: Extract on-the-fly from address_norm in retrieval code.

### 10 CRITICAL: No address-only structural channel
All channels except exact_addr are name-derived. No channel exists for
(country, house_number, street_tokens) independently of name.
Fix: Add structural address blocking channel.

### 11 OK: Channel union is UNION not intersection - correct in all versions.
### 12 OK: country_norm normalization consistent across S1/S2/S3.

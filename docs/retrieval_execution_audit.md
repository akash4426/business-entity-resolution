# Retrieval Execution Audit

This document audits the computational complexity of the baseline and V2 retrieval systems for the Business Entity Resolution project, detailing exactly where computational bottlenecks occurred and how the new architecture avoids them.

## 1. What caused the original computational explosion
The baseline system (V1) performed a Pandas/Polars join across heavily duplicated blocking keys (such as `exact_name`). The root cause of the explosion is that some generic keys (e.g., "Subway", "Starbucks", "Restaurant") appear thousands of times in the dataset. A standard inner join creates a Cartesian product for each matching key.

If "Restaurant" appears 10,000 times in Source 1 and 20,000 times across Source 2 and 3, a single join on that name generates **200,000,000 candidate pairs** just for that one generic string. Accumulating these across the dataset quickly exhausts RAM and forces the OS to OOM kill the process.

## 2. Exact operation responsible
The exact operation responsible was the Polars `.join()` in `src/retrieval/candidate_generation.py` where exact matches and fuzzy overlaps were collected. To prevent the OOM, the V1 authors placed an aggressive `filter_freq(max_freq=15)` cap. This prevented the Cartesian explosion by completely discarding any name/address that appeared more than 15 times, which directly caused the catastrophic 52.6% candidate recall (discarding all legitimate large business chains).

## 3. How the new architecture prevents it
The new Retrieval V2 architecture transforms the execution from a monolithic matrix comparison to a chunked, multi-channel inverted index retrieval built strictly within the C++ Polars backend:

1. **Selective Column Projection**: Only columns required for blocking are loaded, preserving gigabytes of memory.
2. **Frequency Calibrated Caps**: Instead of capping at `15`, the system now dynamically aggregates Document Frequency (DF) across S2 and S3 for keys and tokens. 
3. **Multi-channel Pruning**:
   - `exact_name` and `exact_address` are capped safely at `1,000`.
   - `name_token` (single words) is capped safely at `1,000`.
   - Intersection channels like `address_numbers + name_token` and `postal_code + name_token` have NO caps (or huge caps like 100,000) because the intersection of two distinct signals is mathematically sparse by definition, avoiding the Cartesian explosion entirely while recovering missing variants.
4. **Union then Deduplicate**: Each channel operates independently. The resulting sets are concatenated, deduplicated via `group_by`, and aggregated by provenance (channel tracking) into a final sparse set, avoiding Python Dictionary overhead entirely.

## 4. Naive vs Actual Complexity

- **Test Scale Source Sizes**: S1 = 1.73M, S2 = 4.89M, S3 = 5.08M
- **Naive Comparisons**: 1,732,382 * (4,893,014 + 5,037,056) = **17.2 Trillion pairs**

Using the 20% validation split (346k S1 records):
- **Naive Validation Comparisons**: ~3.44 Trillion pairs
- **Actual Candidate Comparisons (V2)**: 93,509,005 pairs
- **Comparison Reduction Factor**: ~36,793x reduction (99.997% of the comparison space pruned)

## 5. Development Safety Limits
The candidate execution never materializes a full `S1 x S2` representation. The total pair space generated during the validation retrieval phase peaks at roughly 93.5 million records, comfortably utilizing ~4-5 GB of RAM. Memory spikes are managed through active `gc.collect()` passes between channel constructions.

## 6. Channel Impact
- **Channels Included**: Exact Name, Exact Address, Name Token, Token + Address Num, Token + Postal Code.
- **Channels Excluded**: Full TF-IDF matrix similarity, Full dense embedding cosine similarity. Both were excluded as they violate the memory constraints and fail to provide the sparse lookup guarantees required to prevent OOM without aggressive pre-filtering.

# Retrieval V2 Audit Report

## A. Union Semantics
- **Finding**: Retrieval channels are currently combined using `pl.concat` which correctly implements `UNION` (OR) semantics. 
- **Status**: PASS

## B & C. Pre-Union Capping and Hidden Global Caps
- **Finding**: There is a severe hidden recall limit in Channel 1 (Exact Name) and Channel 2 (Address Numbers). The `filter_freq` function drops any blocking key that appears more than 15 times in the dataset. 
- **Impact**: If a business is a chain with > 15 locations (e.g., Subway, McDonald's), it is completely discarded from Channel 1. It must rely entirely on Channel 3 (Name + Address exact match). If there is even a minor formatting difference in the address for that chain location, the true match will be entirely missed. This explains why Candidate Recall is stuck at ~52%.
- **Status**: FAIL (Severe recall destruction due to aggressive frequency capping to avoid OOM).

## D. Country Normalization
- **Finding**: Country codes are currently used as-is. There is no fallback for missing countries or casing/whitespace normalization specifically applied to the `country` column in the retrieval script. If countries are formatted differently, the `inner` join will miss them.
- **Status**: FAIL

## E. Searching Both S2 and S3
- **Finding**: Every channel explicitly performs inner joins on both S2 and S3.
- **Status**: PASS

## F. Candidate ID Collision Protection
- **Finding**: S2 and S3 IDs possess distinct prefixes (`S2-`, `S3-`). The candidate generation simply unions them under `candidate_entity_id`. There are no collisions.
- **Status**: PASS

## G. Actual Candidate Set Fed to Matcher
- **Finding**: The output of `candidate_generation.py` is written to `artifacts/candidates.parquet` and is accurately consumed by the pairwise feature generation and decision engine. 
- **Status**: PASS

## H. Silent Dropping Between Stages
- **Finding**: The pairwise feature script joins candidate pairs back to the raw S1 and S2/S3 dataframes. If an S1 or candidate ID somehow went missing, it would create nulls, but since we start from the candidates dataframe and do `how="left"`, we don't drop rows.
- **Status**: PASS

## Immediate Action Plan
We must redesign retrieval to avoid OOM *without* aggressively discarding queries. Instead of strict `filter_freq` discarding, we will use TF-IDF token blocking, inverted indices for character n-grams, and multi-key address blocking (e.g., zip + country) which naturally restricts candidate explosion without dropping legitimate chain locations.

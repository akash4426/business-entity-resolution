# Experiments Log

This document tracks experimental interventions and architectural trials to improve candidate retrieval and match precision without violating computational constraints on the target hardware.

## Experiment 1: Inverse Document Frequency Blocking (Retrieval V2)
- **Hypothesis**: Replacing static, arbitrary frequency caps (e.g. dropping all entities with a frequency > 15) with dynamic token-based document frequency thresholds (cap=1000) will drastically improve recall for large multi-national/franchise chains without causing memory exhaustion.
- **Implementation**: See `candidate_generation_v2.py`. All blocking keys (exact name, address) and sparse unigrams are dynamically scored for DF across S2 and S3 before being joined.
- **Status**: **SUCCESS**. Candidate pair recall improved from 52.6% to ~79.7%. Comparison space safely reduced by 36,700x (no Cartesian explosion).

## Experiment 2: Semantic Intersection Blocking
- **Hypothesis**: We can safely retrieve generic, high-frequency tokens (like "Bank", "Restaurant", DF > 10,000) *if* we intersect them with a noisy but orthogonal semantic signal (like a building number or a postal code).
- **Implementation**: Added Channels 4 and 5 in `candidate_generation_v2.py` which un-cap token joins only when `address_numbers` or `postal_code` intersect precisely.
- **Status**: **SUCCESS**. Safely expanded candidate net, contributing to the final candidate pool without causing the memory limit crash observed when un-capping standard single tokens.

## Experiment 3: Single Token Threshold Relaxation (Cap = 10,000)
- **Hypothesis**: Allowing a higher DF cap of 10,000 for single `name_tokens` would catch generic variations missed by the 1,000 cap.
- **Status**: **FAILED (OOM)**. The intersection of tokens with 10k frequencies generated >100 Million pairwise intersections per token, instantly exhausting the 16GB RAM limit on the M2. Threshold was reverted to 1,000.

## Future Recommended Experiments
1. **Name Bi-Grams / Token Pairs**: Instead of single tokens, emit combinations of two non-stopword tokens (e.g. `(global, solutions)`). These bi-grams will have massively lower Document Frequencies than single words and can be safely un-capped for near 100% precision blocking.
2. **Character N-Gram Locality Sensitive Hashing (LSH)**: MinHash on character trigrams to safely recover candidates with multiple severe typos that don't share any whole words.
3. **Sparse TF-IDF Candidate Pruning**: If we over-generate candidates (e.g., > 500 per S1), implement a fast dot-product over sparse TF-IDF vectors using Scipy sparse matrices to prune candidates down to top-50 *before* executing expensive string edit distances.

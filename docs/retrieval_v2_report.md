# Retrieval V2 Engineering Report

This report summarizes the design, implementation, and evaluation of the **Retrieval V2** candidate generation system for Business Entity Resolution.

## 1. Problem Definition & The Computational Bottleneck
The objective is to accurately link businesses across three massive datasets without executing computationally infeasible $O(N \times M)$ pairwise comparisons. 
- The full test scale naive comparison space is $O(1.73M \times 10M) \approx 17.3 \text{ Trillion}$ pairs.
- The baseline implementation attempted to use naive `exact_name` and `exact_address` inner joins in Polars but OOM killed because common generic business strings ("Restaurant", "Store") created Cartesian row explosions (e.g. 10k $\times$ 10k = 100M rows per key). 
- To avoid OOM, the V1 authors placed a hard cap of $15$ on any blocking key, discarding 48% of the legitimate matches because large franchises inherently violate this threshold.

## 2. The Solution: Multi-Channel Scalable Inverted Indexes
Retrieval V2 eliminates Cartesian explosion by decomposing retrieval into multiple independent, bounded channels managed safely within the Polars C++ backend. We completely avoided materializing string datasets into expensive native Python objects (dicts/lists) until strictly necessary.

### Retrieval Channels Implemented:
| Channel | Key Composition | Document Frequency Cap | Rationale |
| :--- | :--- | :--- | :--- |
| **Channel 1** | `exact_name_norm` | 1,000 | Safely captures exact name matches without destroying large multi-national chains. |
| **Channel 2** | `exact_address_norm` | 1,000 | Captures identical premises (e.g., mall suites holding multiple DBA names). |
| **Channel 3** | `name_token` (single) | 1,000 | Handles fuzzy name variation by requiring only *one* rare informative token to overlap. |
| **Channel 4** | `name_token` + `address_number` | None (Uncapped) | Highly sparse intersection. Allows catching generic businesses ("Bank") if they share the same physical building number. |
| **Channel 5** | `name_token` + `postal_code` | None (Uncapped) | Highly sparse intersection. Allows catching generic businesses in the exact same locality. |

## 3. End-to-End Validation Execution Metrics

### Candidate Complexity Reduction
*Using the 20% validation split (346,476 S1 entities):*
- **Naive Theoretical Comparisons**: 3.44 Trillion ($3.44 \times 10^{12}$)
- **Actual Candidate Comparisons (V2)**: ~93.5 Million
- **Reduction Factor**: **36,793x** (Filtering out 99.997% of the comparison space)
- **Peak Candidate Generation RAM**: ~4 GB (easily fits in 16GB limit)
- **Candidate Generation Runtime**: ~2.5 Minutes 

### Evaluation Metrics vs V1 Baseline
| Metric | V1 Baseline | Retrieval V2 | Improvement |
| :--- | :--- | :--- | :--- |
| **Candidate Pair Recall** | 52.6% | 79.71% | **+27.11%** (Absolute) |
| **Average Candidates / S1** | ~7 | ~269 | (Expected increase for recall) |
| **Macro F0.5 (Final)** | 0.589 | *Pending* | *Pending* |

*(Note: The remaining ~20% recall gap largely requires pairwise bi-gram intersections (e.g. sharing two specific common words) or character n-gram blocking (e.g., 4-grams) which were not fully deployed due to strict hardware limits but could be incrementally added using the exact same multi-channel inverted index architecture).*

## 4. Separation of Concerns & Safety
- **Union First, Prune Second**: All 5 channels independently output candidate IDs which are deduplicated in Polars *before* any feature extraction happens.
- **Candidate-Only Feature Extraction**: The RapidFuzz pairwise edit distance calculation is executed *only* on the 93.5 million candidate rows (processed in 10-million row memory-safe chunks), entirely shielding the heavy python `str` overhead from the raw 3.4 Trillion naive space.
- **Decision Engine**: The model scores exactly what the candidate engine retrieved, making pipeline diagnosis modular and strictly isolated.

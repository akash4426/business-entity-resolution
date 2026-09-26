# Error Analysis

This document outlines the failure modes currently affecting the end-to-end Entity Resolution pipeline.

## 1. Missing True Candidates (Retrieval Failure)
The single biggest source of error remains the 20% of ground-truth matches that are not retrieved by the Retrieval V2 engine.

**Causes of missing candidates:**
- **Severe Orthographic Variance**: The source names share no exact unigrams (e.g., `MacDonalds` vs `Mcdonalds` where tokenizer fails to split properly, or extreme abbreviation `Intl` vs `International`).
- **Complete Information Mismatch**: Source 1 contains only the legal corporate owner name (e.g. "Yum Brands") while Source 2 contains the DBA trade name (e.g. "KFC") at a slightly shifted mall-suite address.
- **DF Cap Exclusion**: If a business is named "Global Services" and shares no other information, "Global" and "Services" both exceed the `1,000` DF cap and are thus blocked from joining to prevent Cartesian OOMs. Since neither token is rare, and neither the name nor address exactly match, the candidate is never retrieved.

**Mitigation Plan:**
- Token Bi-gram inverted indices (to catch "Global Services" without hitting DF caps).
- Local Locality Sensitive Hashing (LSH) on character n-grams.

## 2. Matcher False Positives
The LightGBM matcher evaluates retrieved candidates based on exact matches and RapidFuzz edit distances.

**Causes of False Positives:**
- **Franchise Over-Collapse**: Two different "Starbucks" in the same exact country might have similar strings in their address fields (e.g. both in "Main St" in different cities, but city boundaries are messy). The model over-indexes on string similarity and mistakenly links them.
- **Generic Holding Companies**: Many shell or holding companies share identical generic names and are registered at the same exact legal agent address (e.g. Corporation Trust Center in Delaware). The model cannot distinguish them because structurally, textually, and physically they appear identical. 

**Mitigation Plan:**
- Introduce a strict Locality check (e.g., Haversine distance if coordinates exist, or exact Postal Code agreement).
- Generate "Hard Negatives" during training (explicitly providing the model with non-matching entities that share the same address, or the same name).

## 3. The 0 / 1 / MANY Decision Engine Errors
The decision engine uses a flat probability threshold (e.g. >0.5) to decide if a candidate is a match.
- This creates threshold instability. If the model is under-confident on a true large chain, it predicts 0 matches.
- We must optimize specifically for Macro F0.5 per S1 by tuning the confidence threshold dynamically, potentially employing a `Top-K if > threshold` logic.

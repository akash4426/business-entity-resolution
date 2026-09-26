# Business Entity Resolution: Methodology

## 1. Problem
The task is to resolve entity records from Source 1 against entities in Source 2 and Source 3. Given a Source 1 entity, we must identify all its matches in Source 2 and Source 3 (0, 1, or many matches). The metric is Macro-average F0.5.

## 2. Data
- 3 sources provided in tab-separated format (TSV).
- Training set: ~2.2M Source 1 entities.
- Test set: ~1.7M Source 1 entities.
- Millions of candidates in S2 and S3 across multiple countries.

## 3. EDA Findings
- S1 represents reference entities.
- S2 and S3 contain noise, missing addresses, abbreviations, and varied legal suffixes.
- Entities are predominantly from the US and India in training. France appears entirely unseen in the test split.
- Source IDs are distinct by prefix (S1-, S2-, S3-).
- Ground truth shows a distribution of zero, singleton, and multi-matches.

## 4. Leakage-Safe Split
- Training data for S1 was randomly split into 80% (train) and 20% (validation) using a deterministic random seed (42).
- Ground-truth links were partitioned by the S1 entity ID.
- No S1 entity overlaps between training and validation.

## 5. Normalization
- Applied multi-view normalization for both business name and address.
- Used Unicode transliteration (`unidecode`) to handle non-Latin characters (e.g., Devanagari in S2).
- Stripped legal suffixes (`inc`, `llc`, `pvt ltd`), domain suffixes (`.com`), and common address abbreviations (`st`, `ave`, `blvd`).
- Extracted numeric tokens from addresses to serve as a high-precision blocking key.

## 6. Candidate Generation
- To avoid Cartesian explosion (e.g., 2.2M S1 × 10M S2/S3), candidates were retrieved using deterministic blocking channels via inner joins in Polars.
- **Channel 1**: Exact Normalized Name + Country match.
- **Channel 2**: Extracted Address Numbers + Country match.
- High-frequency blocking keys (appearing > 500 times) were filtered out to prevent massive candidate explosion from generic tokens.
- Retrieved candidate lists were unioned and deduplicated. Provenance (which channel retrieved the candidate) was retained as a categorical feature.

## 7. Pairwise Features
For each candidate pair retrieved, the following features were computed:
- `exact_name_match` (Boolean)
- `exact_address_match` (Boolean)
- `country_match` (Boolean)
- `retrieved_by_name` (Boolean from provenance)
- `retrieved_by_address` (Boolean from provenance)
- `name_fuzz_ratio` (RapidFuzz edit distance ratio between normalized names)
- `addr_fuzz_ratio` (RapidFuzz edit distance ratio between normalized addresses)

## 8. Hard Negatives
- Candidate generation implicitly acts as a hard negative miner. By joining on exact name or address, the retrieved candidates that are NOT true matches represent the most difficult edge cases (e.g., same name but different address, same address but different business).

## 9. Baseline & LightGBM Model
- Instead of simple Logistic Regression, we opted for **LightGBM**, which excels at tabular pairwise feature evaluation.
- Trained on the pairwise feature vectors using binary logloss.
- Evaluated on the 20% validation split with early stopping to prevent overfitting.
- The model outputs a probability `[0, 1]` representing the likelihood of the pair being a true match.

## 10. Calibration & Decision Engine
- **Thresholding**: Candidate pairs with a predicted probability ≥ 0.5 are classified as matches. 
- **0/1/MANY Strategy**: An S1 entity can retain any number of candidates that pass the threshold. If no candidates pass, the S1 entity is correctly assigned 0 matches.
- We did not enforce strict 1-to-1 global consistency because the problem definition inherently permits a single S1 entity to legitimately map to multiple entities across S2/S3.

## 11. Final Pipeline
- The entire pipeline executes sequentially: Data Loader → Preprocessing/Normalization → Parquet Serialization → Candidate Retrieval → Feature Engineering → LightGBM Inference → Thresholding → TSV Output.
- Leveraged Polars streaming execution (`scan_parquet`) to constrain memory usage well within the 16GB limit.

## 12. Limitations & Future Work
- Phonetic matching (Soundex, Metaphone) was not implemented due to runtime constraints over 10M entities.
- Contextual Embeddings (e.g., MiniLM) could improve candidate recall but would require substantial hardware/time budgets not fitting within standard local constraints.
- Advanced global ownership assignment could marginally improve precision.

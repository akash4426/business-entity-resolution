# EDA Report — Business Entity Resolution Challenge

## 1. Dataset Overview

| Dataset | Rows | Unique IDs | Dup IDs | Fully Dup Rows |
|---------|------|-----------|---------|----------------|
| train_source1 | 2,206,821 | 2,206,821 | 0 | 0 |
| train_source2 | 5,034,616 | 5,034,616 | 0 | 0 |
| train_source3 | 5,285,603 | 5,285,603 | 0 | 0 |
| test_source1 | 1,732,544 | 1,732,544 | 0 | 0 |
| test_source2 | 4,887,273 | 4,887,273 | 0 | 0 |
| test_source3 | 5,082,316 | 5,082,316 | 0 | 0 |

**Columns:** `entity_id`, `business_name`, `business_address`, `country`

**Ground Truth:** 2,206,821 rows, matching S1 entities to S2/S3.

## 2. Data Quality Findings

### Missing Values

| Dataset | Missing Names | Missing Addresses | Missing Country |
|---------|--------------|-------------------|-----------------|
| train_source1 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| train_source2 | 2 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| train_source3 | 13 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| test_source1 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| test_source2 | 46 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| test_source3 | 59 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

### Duplicate Names

Names appearing more than once (exact string match) exist across all sources.
This indicates generic business names that could cause false positive matches.

### Character/Unicode Analysis

- Devanagari script names are prevalent in Source 2 (Indian entities)
- Source 3 contains web-domain style names (e.g., `wilfordhancock.com`)
- Accented Latin characters appear in test data (French entities)
- Some names start with special characters (`<<`, `--`)

## 3. Name Noise Findings

Key patterns observed:
- **Legal suffix variations:** Inc/Incorporated, Corp/Corporation, Ltd/Limited, Pvt/Private, LLC, LLP
- **Punctuation:** `&` vs `and`, periods, commas, apostrophes
- **Capitalization:** Mixed case in S1, ALL CAPS common in S2
- **DBA/Trade names:** Some records contain DBA patterns
- **Web-domain names:** `.com`, `.net` suffixes in S3
- **Transliteration:** Devanagari ↔ Latin script for Indian entities
- **Special prefixes:** `<<`, `--` appear as name prefixes in some sources

## 4. Address Noise Findings

Key patterns observed:
- **Abbreviations:** St/Street, Rd/Road, Ave/Avenue, Blvd/Boulevard
- **State format:** S1 uses 2-letter abbreviations (TX), S3 uses full names (Texas)
- **Component reordering:** S2 sometimes puts city/state before street
- **Landmark references:** `Near`, `Opp.`, `Behind` common in Indian addresses
- **Land references:** `KH NO`, `KHASRA` in Indian source data
- **Missing addresses:** Significant portion of S3 has empty addresses
- **ALL CAPS:** S2 addresses are frequently fully capitalized

## 5. Ground Truth Statistics

- **Total S1 entities:** 2,206,821
- **Total positive links:** 7,638,365
- **Zero-match (singleton):** 123,247 (5.58%)
- **Single-match:** 119,157 (5.40%)
- **Multi-match (>1):** 1,964,417 (89.02%)

### Match Distribution

| Matches | Count | Percentage |
|---------|-------|------------|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

### S1→S2 / S1→S3 Breakdown

- S1→S2 links: 3,693,619
- S1→S3 links: 3,944,746
- S1 with S2 match: 1,919,076 (86.96%)
- S1 with S3 match: 1,940,545 (87.93%)
- S1 with BOTH: 1,776,047 (80.48%)

## 6. Multi-Match / Uniqueness Analysis

- **S2 IDs linked to multiple S1 IDs:** 0
- **S3 IDs linked to multiple S1 IDs:** 0

**No S2/S3 entity maps to multiple S1 entities.**
This means each S2/S3 record belongs to at most one S1 entity.
A uniqueness constraint *could* be justified but should be validated on test data.

## 7. Country Analysis

- **Training countries:** ['India', 'US']
- **Test countries:** ['France', 'India', 'US']
- **Unseen in test:** ['France']
  - `France`: 1,694,445 test records with zero training examples

### Match Rate by Country (Training)

| country   |   total |   zero_match |   has_match |   avg_matches |   zero_match_pct |   has_match_pct |
|:----------|--------:|-------------:|------------:|--------------:|-----------------:|----------------:|
| India     |  883188 |        49351 |      833837 |       3.46454 |          5.58783 |         94.4122 |
| US        | 1323633 |        73896 |     1249737 |       3.45906 |          5.58282 |         94.4172 |

## 8. True-Match Characteristics

Based on 36,703 sampled true-match pairs:

- Name exact match: 4.59%
- Name normalized match: 15.52%
- Country agreement: 100.00%
- Mean name token overlap: 0.556
- Mean address token overlap: 0.523
- Mean name length difference: 3.8 chars
- Mean address length difference: 10.5 chars

### Difficulty Breakdown (True Pairs)

- Easy (name overlap > 0.8): 21.0%
- Medium (0.3–0.8): 58.1%
- Hard (< 0.3): 20.9%

## 9. Key Risks for Phase 2

1. **High singleton rate** — large % of S1 entities have no matches; correctly identifying these is crucial for F0.5
2. **Transliteration challenge** — Devanagari ↔ Latin name matching for Indian entities
3. **Unseen country** — France appears only in test; pipeline must be language-agnostic
4. **Generic names** — common business names appear frequently, requiring address-based disambiguation
5. **Missing addresses** — significant portion of records lack addresses, limiting matching signals
6. **Source-specific formatting** — each source has distinct normalization patterns
7. **Scale** — millions of records require efficient blocking/candidate generation
8. **Multi-match complexity** — many S1 entities match multiple S2/S3 records

## 10. Phase 2 Implications

- Normalization pipeline must handle: legal suffixes, &/and, punctuation, case, Devanagari, French accents
- Blocking strategy must balance recall (many potential pairs) with efficiency (millions of records)
- Feature engineering should include: name similarity, address similarity, country match, length features
- Model must handle unseen country (France) gracefully — no country-specific hardcoding
- Singleton detection should be a dedicated component to maximize F0.5
- Address normalization must handle US state abbrevs, Indian landmarks, French addresses
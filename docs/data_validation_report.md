# Data Validation Report

## 1. Purpose
Validate the integrity, schema, source separation, and ground truth mapping of the Business Entity Resolution dataset.

## 2. Dataset Inventory & Missingness Summary
| Dataset | Total Rows | Unique IDs | Missing IDs | Missing Names | Missing Addresses | Missing Countries | Status |
|---|---|---|---|---|---|---|---|
| train_source1 | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 | PASS |
| train_source2 | 5,034,616 | 5,034,616 | 0 | 1 | 168,967 | 0 | PASS |
| train_source3 | 5,285,603 | 5,285,603 | 0 | 0 | 175,916 | 0 | PASS |
| test_source1 | 1,732,544 | 1,732,544 | 0 | 0 | 0 | 0 | PASS |
| test_source2 | 4,887,273 | 4,887,273 | 0 | 0 | 129,408 | 0 | PASS |
| test_source3 | 5,082,316 | 5,082,316 | 0 | 0 | 136,098 | 0 | PASS |

## 3. Schema & ID Validation
**train_source1** PASSED schema and ID checks.
**train_source2** PASSED schema and ID checks.
**train_source3** PASSED schema and ID checks.
**test_source1** PASSED schema and ID checks.
**test_source2** PASSED schema and ID checks.
**test_source3** PASSED schema and ID checks.

## 4. Source Separation Validation
**Train source separation**: PASS
**Test source separation**: PASS

## 5. Ground-Truth Validation
**Status**: PASS
- Total Rows: 2,206,821
- Unique S1 IDs: 2,206,821
- Empty-match Rows: 123,247
- Non-empty-match Rows: 2,083,574
- Total Referenced S2 IDs: 3,693,619
- Total Referenced S3 IDs: 3,944,746
- Invalid References: 0
- Duplicate References In Row: 0

## 6. Country Discovery
| Country | Train Count | Test Count |
|---|---|---|
| US | 7,510,506 | 4,480,137 |
| India | 5,016,534 | 5,527,551 |
| France | 0 | 1,694,445 |

## 7. Validation Checks Implemented
- [x] Source schema validation
- [x] Entity ID presence, uniqueness, and prefix validation
- [x] Source separation (no ID overlap between S1/S2/S3)
- [x] Ground truth schema validation
- [x] Ground truth S1 ID existence in S1 dataset
- [x] Ground truth matched IDs prefix and existence in S2/S3 datasets
- [x] Ground truth empty matches allowed
- [x] Ground truth duplicate matched IDs detection
- [x] Missingness calculation for all sources
- [x] Dynamic country discovery

## 8. Final Status
**Overall Status: PASS**

## 9. Reproduction Command
```bash
cd code/business_entity_resolution && PYTHONPATH=. python src/run_validation.py
```
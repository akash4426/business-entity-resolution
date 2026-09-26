# First Submission Diagnostics

## Current Submission
The pipeline uses exact name and exact address blocking channels, yielding ~0.609 validation Macro F0.5.

## Validation Dataset
- Total S1 Entities: 441364
- Total Ground Truth Links: 1527298

## Candidate Recall
- Pair Recall (Links): 0.5260
- Entity Recall (All true matches present): 0.2545
- At-least-one Recall: 0.8387

## Candidate Size
- Mean: 5.40
- Median: 3.00
- P90: 14.00
- P95: 19.00
- Max: 56

## Model Performance
- Macro F0.5: 0.6097
- Macro Precision: 0.7052
- Macro Recall: 0.4715
- Total TP: 705028, Total FP: 125614, Total FN: 822270

## Actual vs Predicted Cardinality
| Match Count | Actual | Predicted |
|-------------|--------|-----------|
| 0 | 24585 | 91244 |
| 1 | 23898 | 110931 |
| 2 | 75096 | 103902 |
| 3 | 106409 | 70917 |
| 4 | 96594 | 37887 |
| 5 | 64284 | 16914 |
| 6+ | 50498 | 9569 |

## Retrieval vs Model vs Decision Errors
- **Retrieval Failure**: 74.55% (True match missed by candidates)
- **Model Failure**: 11.83% (True match in candidates but incorrectly scored)
- **Decision Failure**: 0.00% (Thresholding issues)
- **Correct**: 13.62%

## Top-K Recall (All True Matches)
- Top 1: 0.0268
- Top 5: 0.1914
- Top 10: 0.2091
- Top 50: 0.2105

## Main Bottleneck
**RETRIEVAL** is the primary bottleneck.

## Recommended Next Experiment
Focus on improving the RETRIEVAL phase.

# Business Entity Resolution — Step-by-Step Diagram & Team Execution Plan

---

## PART 1 — CLEAR STEP-BY-STEP ARCHITECTURE DIAGRAM

Each box = one concrete step. Each arrow = a saved artifact handed to the next step.
🚦 = mandatory gate — **do not proceed past a 🚦 until it passes.**

```
STEP 0 ─────────────────────────────────────────────────────────────
  INSPECT REPO + DATASET
  IN:  raw dataset/train, dataset/test
  DO:  confirm file counts, column schema, row counts match EDA doc
  OUT: confirmed data contract
──────────────────────────────────────────────────────────────────

STEP 1 ─────────────────────────────────────────────────────────────
  DATA CONTRACT + VALIDATION
  IN:  raw TSVs
  DO:  schema check, ID prefix check, dedup check, missingness report,
       ground-truth referential check (do GT IDs exist in S2/S3?)
  OUT: src/data/loader.py, schema.py, validation.py + passing tests
──────────────────────────────────────────────────────────────────

STEP 2 ─────────────────────────────────────────────────────────────
  LEAKAGE-SAFE TRAIN / VALIDATION SPLIT
  IN:  train_source1.tsv, train_ground_truth.tsv
  DO:  split at S1 entity level, fixed seed, no S1 entity in both sides
  OUT: train_split.parquet, val_split.parquet (S1 ID lists only)
──────────────────────────────────────────────────────────────────

STEP 3 ─────────────────────────────────────────────────────────────
  MULTI-VIEW NORMALIZATION
  IN:  raw S1/S2/S3 records
  DO:  build raw / normalized / transliterated / tokens / char n-grams /
       phonetic / structured-address views for name + address
  OUT: normalized_s1.parquet, normalized_s2.parquet, normalized_s3.parquet
       (original columns preserved alongside new ones)
──────────────────────────────────────────────────────────────────

STEP 4 ─────────────────────────────────────────────────────────────
  DYNAMIC COUNTRY SHARDING
  IN:  normalized records
  DO:  partition by country value (no hardcoded list) — India / US /
       France / any future country auto-handled
  OUT: per-country record shards, each processed independently downstream
──────────────────────────────────────────────────────────────────

STEP 5 ─────────────────────────────────────────────────────────────
  RETRIEVAL CHANNEL 1 → 7 (run per country shard)
  5a. Exact-name inverted index
  5b. Token inverted index (common-token capped)
  5c. Character n-gram / sparse TF-IDF retrieval
  5d. Phonetic (Double Metaphone) index
  5e. MinHash LSH near-duplicate blocking
  5f. Sorted Neighborhood (sort key + sliding window)
  5g. Address retrieval (postal / street-number / locality)
  OUT: per-channel candidate lists + provenance flag per channel
──────────────────────────────────────────────────────────────────

STEP 6 ─────────────────────────────────────────────────────────────
  CANDIDATE UNION + DEDUP
  IN:  outputs of Steps 5a–5g
  DO:  union all channels, dedup IDs, store which channel(s) hit,
       store channel-count per pair
  OUT: candidates_lexical.parquet
──────────────────────────────────────────────────────────────────

🚦 STEP 7 ─────────────────────────────────────────────────────────
  CANDIDATE RECALL GATE #1 (lexical only)
  DO:  compare candidates_lexical vs ground truth on val split
  PASS IF: recall meets target (define target, e.g. ≥97%) across
       country / 0-1-many / easy-medium-hard breakdowns
  IF FAIL → go back to Step 5, add/tune channels. DO NOT CONTINUE.
──────────────────────────────────────────────────────────────────

STEP 8 ─────────────────────────────────────────────────────────────
  MULTILINGUAL EMBEDDING RETRIEVAL
  IN:  normalized records (Step 3)
  DO:  embed name / address / combined, build FAISS IVF-PQ index
       (quantized, not flat), retrieve top-K per query
  OUT: candidates_embedding.parquet
──────────────────────────────────────────────────────────────────

🚦 STEP 9 ─────────────────────────────────────────────────────────
  CANDIDATE RECALL GATE #2 (lexical + embedding)
  DO:  union Step 6 + Step 8, re-measure recall
  PASS IF: recall improves to target and P95 candidate count stays
       within compute budget
  IF FAIL → tune embedding K / index / add channels. DO NOT CONTINUE.
──────────────────────────────────────────────────────────────────

STEP 10 ────────────────────────────────────────────────────────────
  FREEZE CANDIDATE GENERATION
  DO:  lock the exact config (channels, params, K) that passed Step 9
  OUT: candidate_pairs.tsv generation code is now FROZEN — never
       changed again, only re-run identically for test
──────────────────────────────────────────────────────────────────

STEP 11 ────────────────────────────────────────────────────────────
  CASCADE FEATURE ENGINE — L1 (cheap)
  IN:  frozen candidate pairs
  DO:  exact/normalized match flags, token overlap, channel count
  OUT: pruned candidate set (drop obvious non-matches)
──────────────────────────────────────────────────────────────────

STEP 12 ────────────────────────────────────────────────────────────
  CASCADE FEATURE ENGINE — L2 (expensive, survivors only)
  DO:  edit distance, Jaro-Winkler, phonetic agreement, embedding
       cosine, address component overlap, structural features
  OUT: features.parquet (one row per surviving candidate pair)
──────────────────────────────────────────────────────────────────

STEP 13 ────────────────────────────────────────────────────────────
  LOGISTIC REGRESSION BASELINE
  IN:  features.parquet (train split)
  DO:  train, evaluate Macro F0.5 on val — sanity-checks features/labels
  OUT: baseline_model.pkl, baseline metrics report
──────────────────────────────────────────────────────────────────

STEP 14 ────────────────────────────────────────────────────────────
  HARD NEGATIVE MINING
  IN:  train-split candidates + labels
  DO:  mine near-miss non-matches (same name/address/country but not
       a true match), respecting train/val boundary
  OUT: augmented training set with hard negatives
──────────────────────────────────────────────────────────────────

STEP 15 ────────────────────────────────────────────────────────────
  LIGHTGBM MATCHER
  IN:  augmented training features
  DO:  train, compare vs. LR baseline on val Macro F0.5
  OUT: lightgbm_model.pkl (keep only if it beats baseline)
──────────────────────────────────────────────────────────────────

STEP 16 ────────────────────────────────────────────────────────────
  PROBABILITY CALIBRATION
  DO:  fit Platt + isotonic on train, compare calibration curves on val
  OUT: calibrated_model.pkl
──────────────────────────────────────────────────────────────────

STEP 17 ────────────────────────────────────────────────────────────
  ENTITY-LEVEL 0/1/MANY DECISION ENGINE
  IN:  calibrated probabilities per candidate, grouped by S1
  DO:  sort per S1, apply threshold + margin + evidence-strength rules,
       explicitly branch: no-candidate / weak / one-strong / multi-strong
  OUT: decision_engine.py
──────────────────────────────────────────────────────────────────

STEP 18 ────────────────────────────────────────────────────────────
  F0.5 THRESHOLD / MARGIN OPTIMIZATION
  IN:  decision engine + val set
  DO:  grid/Optuna search over threshold + margin, optimize Macro F0.5
  OUT: final_thresholds.yaml
──────────────────────────────────────────────────────────────────

STEP 19 ────────────────────────────────────────────────────────────
  GLOBAL CONSISTENCY (groupby-argmax) — VALIDATION-GATED
  DO:  for candidate IDs claimed by >1 S1, keep only top-scoring
       assignment above threshold; measure Macro F0.5 with vs. without
  OUT: keep ONLY if it improves val Macro F0.5
──────────────────────────────────────────────────────────────────

STEP 20 ────────────────────────────────────────────────────────────
  SELECTIVE LOCAL LLM — VALIDATION-GATED, BUDGET-CAPPED
  DO:  identify decision-changing ambiguous S1 entities only, cap total
       LLM calls (e.g. ≤50K), run local Qwen2.5-7B/Phi-3-mini, measure
       Macro F0.5 with vs. without
  OUT: keep ONLY if it improves val Macro F0.5, else remove entirely
──────────────────────────────────────────────────────────────────

STEP 21 ────────────────────────────────────────────────────────────
  FRANCE / UNSEEN-COUNTRY ROBUSTNESS EVAL
  DO:  score India / US / France separately on val (simulate via
       held-out-country test), report gaps, patch normalization/
       retrieval if France underperforms
  OUT: docs/error_analysis.md (country section)
──────────────────────────────────────────────────────────────────

STEP 22 ────────────────────────────────────────────────────────────
  ERROR ANALYSIS + ABLATION + SHAP
  DO:  bucket failures, run A0→A12 ablation table, generate SHAP on
       final matcher
  OUT: docs/experiments.md, docs/error_analysis.md
──────────────────────────────────────────────────────────────────

STEP 23 ────────────────────────────────────────────────────────────
  FULL TEST INFERENCE (frozen pipeline, no ground truth touched)
  IN:  dataset/test/*
  DO:  run Steps 3→20 exactly as frozen, over full test set
  OUT: raw predictions
──────────────────────────────────────────────────────────────────

STEP 24 ────────────────────────────────────────────────────────────
  GENERATE OUTPUT FILES
  OUT: output/matching_results.tsv, output/candidate_pairs.tsv
──────────────────────────────────────────────────────────────────

🚦 STEP 25 ─────────────────────────────────────────────────────────
  VALIDATE + ASSERT
  DO:  run utils/validate_submission.py → must PASS
       assert FINAL_MATCHES ⊆ CANDIDATES
  IF FAIL → fix output generation. DO NOT SUBMIT.
──────────────────────────────────────────────────────────────────

STEP 26 ────────────────────────────────────────────────────────────
  DOCUMENTATION + SUBMISSION ZIP
  OUT: README.md, methodology.md, experiments.md, error_analysis.md,
       <team_name>_submission.zip
──────────────────────────────────────────────────────────────────
```

---

## PART 2 — TEAM EXECUTION PLAN

### Suggested team structure (4–5 people)

| Role                                                               | Owns pipeline steps    | Primary skills                                        |
| ------------------------------------------------------------------ | ---------------------- | ----------------------------------------------------- |
| **A. Data & Infra Lead**                                           | 0, 1, 2, 4, 10, 25, 26 | data engineering, validation, packaging, repo hygiene |
| **B. Retrieval Engineer**                                          | 3, 5a–5g, 6, 7, 8, 9   | search/blocking, FAISS, LSH, scalability              |
| **C. ML Modeling Engineer**                                        | 11, 12, 13, 14, 15, 16 | feature engineering, LightGBM, calibration            |
| **D. Decision & Evaluation Lead**                                  | 17, 18, 19, 21, 22     | metrics, threshold tuning, error analysis, ablations  |
| **E. LLM & Docs Lead** (can double with A or D on a 4-person team) | 20, 23, 24, 26         | local LLM serving, prompt design, methodology writing |

If only 4 people: merge **E into D** (Decision/Eval lead also owns the selective LLM, since it plugs directly into the decision engine) and have **A** own final inference + packaging with help from whoever is free.

### Critical path (cannot be parallelized — hard dependencies)

```
Step 1 → Step 2 → Step 3 → Step 4 → Steps 5–9 (🚦 gates) → Step 10 (FREEZE)
   → Step 11–12 → Step 13 → Step 15 → Step 16 → Step 17 → Step 18
   → Step 19 → Step 20 → Step 23 → Step 24 → 🚦 Step 25 → Step 26
```

Everything downstream of Step 10 depends on the frozen candidate set. Everything downstream of Step 16 depends on the calibrated model. **These two freezes are your synchronization points — plan team check-ins around them, not around calendar days.**

### What CAN run in parallel

- **Steps 1–2** (Data/Infra) run alongside early **Step 3 prototyping** (Retrieval) on a small sample, since normalization logic can be drafted before the split is finalized.
- **Steps 5a–5g** (the seven retrieval channels) are largely independent of each other — split across 2 people if available (e.g., B1 does exact/token/character, B2 does phonetic/LSH/SNM/address), then merge at Step 6.
- While Retrieval is working through Steps 5–9, **ML Modeling (C)** can build and unit-test the feature engine (Step 11–12) and LightGBM training code (Step 15) against a _toy/sample candidate set_, so it's ready to run for real the moment Step 10 freezes.
- **Decision & Eval (D)** can build the Macro F0.5 metric implementation and the decision-engine skeleton (Step 17–18) early, independent of retrieval/matching progress, and test it with synthetic scores.
- **Docs (E/A)** can draft `docs/eda_report.md` and the methodology template skeleton from day one, filling in numbers as each stage completes, rather than writing everything at the end.
- Ablations (Step 22) can be run incrementally as each component lands, rather than as one block at the end.

### Suggested timeline (example: 6-day sprint — adjust to your actual deadline)

| Day       | Focus                                                                                                                    | Milestone                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Day 1** | Steps 0–4: data contract, split, normalization, country sharding                                                         | Normalized, sharded data ready for all downstream work                         |
| **Day 2** | Steps 5–7: build all lexical retrieval channels, union, Gate #1                                                          | 🚦 Lexical candidate recall gate passes                                        |
| **Day 3** | Steps 8–10: embedding retrieval, Gate #2, freeze candidates                                                              | 🚦 Candidate generation frozen — **hard sync point, whole team reviews**       |
| **Day 4** | Steps 11–16: features, baseline, hard negatives, LightGBM, calibration                                                   | Calibrated matcher beats baseline on val Macro F0.5                            |
| **Day 5** | Steps 17–22: decision engine, threshold tuning, global consistency, selective LLM, France eval, error analysis/ablations | Final validation Macro F0.5 locked in; methodology draft complete              |
| **Day 6** | Steps 23–26: full test inference, output generation, validator, docs, ZIP                                                | 🚦 Validator PASS, submission ready with time to spare for a leaderboard check |

Build in slack: submit a rough end-to-end run (even with weak components) by end of **Day 3**, so you have a working submission before you start optimizing. This protects you from a broken pipeline the night before the deadline.

### Team workflow rules

1. **One shared repo, feature branches per pipeline stage**, PR review before merging into `main` — especially around the two freeze points (candidate generation, calibrated model), since everyone downstream depends on them being stable.
2. **Config-driven handoffs**: nobody hardcodes parameters in their module — every tunable value lives in the relevant `configs/*.yaml`, so another teammate can adjust it without touching code.
3. **Daily 15-minute sync** focused on exactly three questions: _what's blocking me, what did my last gate/metric show, what do I need from someone else's output._
4. **No one touches test data until Step 23.** All development, tuning, and debugging happens against train/validation only — enforce this as a hard team rule, not just a design note.
5. **Whoever hits a 🚦 gate posts the numbers to the team** (recall %, Macro F0.5, breakdown by country/difficulty) before anyone proceeds — gates are team checkpoints, not solo sign-offs.
6. **Methodology doc is written incrementally**, one section per completed stage, by whoever owns that stage — not reconstructed from memory at the end.
7. **If a component doesn't beat its baseline on validation Macro F0.5, cut it** — no one keeps a personally-built component "because it took effort." Decisions are validation-driven, not sentiment-driven.

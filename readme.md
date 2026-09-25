# Business Entity Resolution — Final Architecture & Execution Plan

**Project:** hybrid-entity-resolution
**Metric:** Macro F0.5 (precision-weighted, per-Source-1-entity average)
**Scale:** Train ~2.2M S1 / 5.0M S2 / 5.3M S3 · Test ~1.7M S1 / 4.9M S2 / 5.1M S3
**Constraint:** Offline only, no external lookups, final models MIT/Apache-2.0, ≤8B params
**Hardware target:** MacBook Air M2, 16GB RAM, no GPU assumed

---

## 1. Core Design Principle

> **Separate retrieval from matching from decision from consistency.**

- **Retrieval** answers: _who could possibly be the same business?_ (optimize recall)
- **Matching** answers: _how strong is the evidence these two records are the same entity?_ (calibrated probability)
- **Decision** answers: _which 0, 1, or many candidates should actually be returned?_ (optimize precision under F0.5)
- **Global consistency** answers: _are the resulting assignments mutually consistent given the data's structure?_
- **Evaluation** answers: _did each component actually earn its place?_

**Guiding rule:** if a component doesn't measurably improve validation Macro F0.5, remove it. Complexity is not the goal — recall, precision, F0.5, scalability, generalization to France, and reproducibility are.

---

## 2. Key EDA Facts Driving Design

| Fact                                                                         | Design implication                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 89% of S1 entities have multi-match (2–10 matches), only 5.4% have exactly 1 | **Never use top-1 matching.** Decision engine must natively support 0/1/many.                                                                                                                                       |
| 100% country agreement in true matches                                       | Country is a **hard blocking key**, not just a feature — shard the entire pipeline by country.                                                                                                                      |
| Zero S2/S3 IDs linked to >1 S1 entity in training                            | Structural invariant: **each S2/S3 record belongs to at most one real business.** Exploit this for cheap global conflict resolution — don't just "experiment" with it, expect it to help and validate that it does. |
| France unseen in training, ~1.7M test records                                | Normalization, embeddings, and retrieval must generalize with **zero France-specific tuning**; explicit held-out-country stress test required before trusting the leaderboard result.                               |
| 20.9% of matches are "hard" (low token overlap)                              | Retrieval must include phonetic + embedding channels, not just lexical.                                                                                                                                             |
| Exact name match only in 4.6% of true positives                              | Exact/normalized matching alone is provably insufficient — multi-signal evidence is mandatory.                                                                                                                      |
| All-pairs = ~10M × ~10M comparisons                                          | Infeasible. Retrieval must be sub-quadratic; naive Hungarian assignment (O(n³)) for global consistency is **also infeasible** — replaced with a vectorized groupby-argmax approach.                                 |

---

## 3. Final Architecture

```
                              INPUT DATA (S1, S2, S3, per split)
                                        |
                                        v
                      DATA CONTRACT + VALIDATION (schema, IDs, missingness)
                                        |
                                        v
                LEAKAGE-SAFE S1-LEVEL TRAIN / VALIDATION SPLIT (fixed seed)
                                        |
                                        v
                          MULTI-VIEW NORMALIZATION
        raw | normalized | transliterated | tokens | char n-grams | phonetic | structured address
                                        |
                                        v
                    DYNAMIC COUNTRY SHARDING (India / US / France / future)
                    (bounds memory: pipeline processes one country shard at a time)
                                        |
                                        v
              ┌─────────────────────────────────────────────────────────┐
              │        MULTI-CHANNEL CANDIDATE RETRIEVAL (per shard)      │
              │  1. Exact name (inverted index on normalized name)         │
              │  2. Token retrieval (inverted index, common-token capped)  │
              │  3. Character retrieval (char n-gram / sparse TF-IDF)      │
              │  4. Phonetic retrieval (Double Metaphone index)            │
              │  5. MinHash LSH (near-duplicate shingle blocking)          │
              │  6. Sorted Neighborhood (sort key + sliding window)        │
              │  7. Address retrieval (postal / street-number / locality)  │
              │  8. Multilingual embedding ANN (FAISS IVF-PQ, quantized)   │
              └───────────────────────┬─────────────────────────────────┘
                                        v
                    CANDIDATE UNION + DEDUP + RETRIEVAL PROVENANCE
                (store which channel(s) retrieved each pair — becomes a feature)
                                        |
                                        v
                    ★ CANDIDATE RECALL QUALITY GATE ★ (mandatory checkpoint)
              recall vs ground truth, broken down by country / 0-1-many / difficulty
                    insufficient → STOP, improve retrieval, do not proceed
                                        |
                                        v (sufficient)
                    FREEZE CANDIDATE GENERATION CONFIG
              (candidate_pairs.tsv = this exact frozen set, forever)
                                        |
                                        v
              ┌─────────────────────────────────────────────────────────┐
              │         CASCADE PAIRWISE FEATURE ENGINE                   │
              │  L1 (cheap, vectorized): exact/normalized flags,           │
              │     token overlap, retrieval-channel count                │
              │     → prune obvious non-matches before L2                 │
              │  L2 (expensive, survivors only): edit distance,            │
              │     Jaro-Winkler, phonetic agreement, embedding cosine,    │
              │     address component overlap, structural features        │
              └───────────────────────┬─────────────────────────────────┘
                                        v
                        LOGISTIC REGRESSION BASELINE
                (sanity-checks features/labels, catches leakage early)
                                        |
                                        v
                          HARD NEGATIVE MINING
        (same-name/same-address/high-similarity non-matches, train-split only)
                                        |
                                        v
                          LIGHTGBM MATCHER
                    (compared against LR baseline on Macro F0.5)
                                        |
                                        v
                    PROBABILITY CALIBRATION (Platt / isotonic)
                        (fit on train split only)
                                        |
                                        v
              ┌─────────────────────────────────────────────────────────┐
              │           ENTITY-LEVEL 0 / 1 / MANY DECISION ENGINE        │
              │  per S1: sort candidate probabilities, apply              │
              │  absolute threshold + top/next margin + evidence          │
              │  strength + candidate-count distribution                  │
              │  → explicit handling of: no candidates, weak candidates,   │
              │    one strong, multiple strong, ambiguous                 │
              └───────────────────────┬─────────────────────────────────┘
                                        v
                      F0.5 THRESHOLD / MARGIN OPTIMIZATION
                    (tuned on validation only, documented, not arbitrary)
                                        |
                                        v
              ┌─────────────────────────────────────────────────────────┐
              │         GLOBAL CONSISTENCY (groupby-argmax)                │
              │  Exploits the "≤1 owning S1 per S2/S3 record" invariant:   │
              │  for each candidate ID claimed by multiple S1 entities,    │
              │  keep only the highest-scoring assignment above threshold. │
              │  O(n log n) vectorized groupby — NOT Hungarian assignment  │
              │  (infeasible at this scale). Validation-gated.             │
              │  Never forces one-to-one on the S1 side — one S1 can       │
              │  still own many S2/S3 records.                             │
              └───────────────────────┬─────────────────────────────────┘
                                        v
                    ┌───────────────────────────────┐
                    │  HIGH-CONFIDENCE → direct decision │
                    │  AMBIGUOUS (thin top/next margin,  │
                    │  both candidates otherwise strong, │
                    │  decision-changing) → SELECTIVE     │
                    │  LOCAL LLM, hard call-budget capped │
                    │  (e.g. ≤50K calls on full test set) │
                    └───────────────────┬───────────────┘
                                        v
                                    EVALUATION
        Macro F0.5 · candidate recall · 0/1/many breakdown · India/US/France ·
        ablations · error analysis · SHAP · runtime · memory
                                        |
                                        v
                                FINAL OUTPUTS
                    matching_results.tsv  +  candidate_pairs.tsv
                    (assert FINAL_MATCHES ⊆ CANDIDATES)
```

---

## 4. Why This Is a Standout Solution (not a generic ER demo)

1. **Exploits a discovered structural invariant** (single-owner S2/S3 records) with a _cheap, scalable_ mechanism (groupby-argmax) instead of either ignoring it or reaching for an infeasible optimal-assignment algorithm. This is evidence-driven engineering, not a textbook default.
2. **Country sharding as a first-class architectural decision**, justified by the 100%-country-agreement EDA finding — turns an unbounded-memory problem into N bounded ones, and doubles as the retrieval blocking strategy.
3. **Cascade (L1/L2) feature computation** — avoids paying for expensive similarity computation (embeddings, edit distance) on the ~99% of candidate pairs that a cheap flag already rules out. This is standard practice in large-scale ranking systems and is rarely done properly in challenge submissions.
4. **Retrieval channel diversity matched to the actual noise taxonomy found in EDA** (typos → character/phonetic; transliteration → multilingual embeddings; near-duplicates → MinHash LSH; ordering → Sorted Neighborhood) rather than generic "embeddings + fuzzy match."
5. **LLM used surgically, under an explicit compute budget, gated on decision-impact** (not just score-ambiguity) — and removed entirely if it doesn't move validation Macro F0.5. This shows judgment rather than GenAI-for-its-own-sake.
6. **Explicit unseen-country (France) stress test** built into the validation protocol, not just "handled by not hardcoding" — tests generalization, which is exactly what the hidden test set is designed to probe.
7. **Every claim is validation-gated**: candidate recall gate, calibration check, threshold tuning, global consistency, and LLM usage all require measured improvement before being kept. This produces an honest, defensible methodology document rather than an unfalsifiable narrative.

---

## 5. Tech Stack

| Layer                   | Tools                                                                                                                   | Notes                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Data / normalization    | `pandas`, `polars` (large joins), `numpy`, `unidecode`, `ftfy`, `regex`                                                 | polars for the largest feature-table operations to control memory |
| Address parsing         | `libpostal` (MIT)                                                                                                       | local, rule-based, no external calls                              |
| Phonetic                | `jellyfish` (Double Metaphone)                                                                                          | typo-resilient blocking                                           |
| Near-duplicate blocking | `datasketch` (MinHash LSH)                                                                                              | sub-linear, complements embeddings                                |
| Embeddings              | `sentence-transformers` + `intfloat/multilingual-e5-base` or `BAAI/bge-m3` (verify exact license)                       | multilingual for transliteration + French                         |
| ANN index               | `faiss-cpu`, **IVF-PQ** (quantized), not flat                                                                           | memory-bounded at 10M+ scale                                      |
| Lexical similarity      | `rapidfuzz`                                                                                                             | vectorized, fast Levenshtein/Jaro-Winkler                         |
| Baseline model          | `scikit-learn` LogisticRegression                                                                                       | leakage/label sanity check                                        |
| Matcher                 | `LightGBM` (MIT)                                                                                                        | primary pairwise classifier                                       |
| Calibration             | `scikit-learn` (Platt / isotonic)                                                                                       | fit on train split only                                           |
| Local LLM (selective)   | `Qwen2.5-7B-Instruct` or `Phi-3-mini` (Apache-2.0 / MIT, verify exact license) via `llama.cpp` or `vLLM`, fully offline | capped call budget                                                |
| Global consistency      | `pandas`/`polars` groupby-argmax                                                                                        | replaces infeasible Hungarian assignment                          |
| Threshold tuning        | `Optuna`                                                                                                                | Macro F0.5 objective on validation                                |
| Explainability          | `SHAP`                                                                                                                  | on final LightGBM model                                           |
| Config                  | YAML per stage (`baseline.yaml`, `retrieval.yaml`, `matcher.yaml`, `final.yaml`)                                        | no hardcoded experimental params                                  |
| Testing                 | `pytest`                                                                                                                | schema, normalization, retrieval, decision, output contract       |
| Experiment tracking     | `MLflow` or structured JSON/CSV logs                                                                                    | reproducibility                                                   |

---

## 6. Project Structure

```
dataset/
    train/
    test/

output/
    matching_results.tsv
    candidate_pairs.tsv

notebooks/
    01_eda.ipynb

docs/
    eda_report.md
    methodology.md
    experiments.md
    error_analysis.md

code/
    business_entity_resolution/
        src/
            data/          # loader.py, schema.py, validation.py
            preprocessing/ # multi-view normalization
            retrieval/      # exact, token, char, phonetic, LSH, SNM, address, embedding ANN
            features/       # L1/L2 cascade feature engine
            models/         # LR baseline, LightGBM, calibration
            decision/       # 0/1/many entity decision engine, threshold/margin tuning
            llm/            # selective local LLM re-ranker
            evaluation/      # split.py, metrics.py (Macro F0.5), ablation harness
            config.py
            pipeline.py
        tests/
        configs/
            baseline.yaml
            retrieval.yaml
            matcher.yaml
            final.yaml
        artifacts/          # cached embeddings, indexes, models
        README.md
        requirements.txt

Documentation_template.md
```

---

## 7. Implementation Order (do not skip or reorder)

1. Freeze data contract (schema, ID uniqueness, missingness, ground-truth validity)
2. Leakage-safe S1-level train/validation split (fixed seed)
3. Multi-view normalization (name + address, all views, deterministic and tested)
4. Dynamic country sharding (no hardcoded country list)
5. Exact-name retrieval → measure
6. Token retrieval → measure
7. Character retrieval → measure
8. Phonetic retrieval → measure
9. MinHash LSH retrieval → measure
10. Sorted Neighborhood retrieval → measure
11. Address retrieval → measure
12. Candidate union + provenance
13. **Candidate recall gate #1** (lexical channels only) — stop and fix if weak
14. Multilingual embedding ANN retrieval (IVF-PQ)
15. **Candidate recall gate #2** (with embeddings) — stop and fix if weak
16. **Freeze candidate generation** — this exact config produces `candidate_pairs.tsv`
17. Cascade pairwise feature engine (L1 prune → L2 full features)
18. Logistic regression baseline (sanity check, Macro F0.5)
19. Hard negative mining (train split only)
20. LightGBM matcher (compare vs. LR on Macro F0.5)
21. Probability calibration (Platt vs. isotonic, train-fit only)
22. Entity-level 0/1/many decision engine
23. F0.5 threshold + margin optimization (validation only, documented)
24. Global consistency via groupby-argmax — validation-gated
25. Selective local LLM re-ranker — validation-gated, budget-capped
26. France / unseen-country robustness evaluation
27. Error analysis (bucketed failure modes with root cause + examples)
28. Ablation study (A0 baseline → A12 full system, each component measured)
29. SHAP / feature importance on final matcher
30. Full frozen-pipeline test inference (no ground truth, no external lookup, no hand-tuning to test)
31. Generate `matching_results.tsv`
32. Generate `candidate_pairs.tsv`
33. Run official validator (`utils/validate_submission.py`) — must PASS
34. Internal assertion: `FINAL_MATCHES ⊆ CANDIDATES`
35. Run full test suite
36. Write `docs/methodology.md`, `docs/experiments.md`, `docs/error_analysis.md`
37. Prepare final submission ZIP

---

## 8. Engineering Rules (non-negotiable)

- No all-pairs comparison, anywhere, at any stage.
- No Hungarian / O(n³) assignment at this scale — global consistency uses groupby-argmax.
- Process per country shard to bound memory; never load all three sources fully joined at once.
- Cache: normalized fields, embeddings, ANN indexes, candidate sets, trained models — all resumable.
- Fit TF-IDF, vocabularies, calibration, and thresholds **only on the training split**. Validation is for tuning/evaluation; test is inference-only.
- `candidate_pairs.tsv` must be the _exact_ frozen candidate set fed to the matcher — never regenerate a different one for final inference.
- Any model (LightGBM, embedding model, LLM) must have its **exact model license** verified as MIT/Apache-2.0 and ≤8B params — do not infer this from the library's license.
- If a component doesn't improve validation Macro F0.5, remove it — do not keep it for appearances.

---

## 9. Definition of Done

- [ ] Dataset contract validated; original files untouched
- [ ] Leakage-safe S1-level validation split
- [ ] Multi-view normalization (name + address) implemented and tested
- [ ] Dynamic country sharding (no hardcoded countries)
- [ ] All retrieval channels implemented: exact, token, character, phonetic, MinHash LSH, Sorted Neighborhood, address, embedding ANN
- [ ] Candidate union + provenance stored
- [ ] Candidate recall measured and gated (pre- and post-embedding)
- [ ] Candidate generation frozen before matcher training
- [ ] Cascade (L1/L2) pairwise feature engine
- [ ] Logistic regression baseline evaluated
- [ ] Hard negative mining implemented
- [ ] LightGBM matcher trained and compared against baseline
- [ ] Calibration evaluated (Platt vs. isotonic)
- [ ] 0/1/many entity decision engine implemented (no top-1 forcing)
- [ ] Macro F0.5 threshold/margin optimization completed on validation
- [ ] Global consistency (groupby-argmax) implemented and validation-gated
- [ ] Selective local LLM implemented, budget-capped, validation-gated (kept only if it helps)
- [ ] France evaluated separately from India/US
- [ ] Error analysis completed with bucketed failure modes
- [ ] Full ablation study (A0–A12) completed
- [ ] SHAP / feature importance generated
- [ ] Full test inference run on frozen pipeline
- [ ] `matching_results.tsv` and `candidate_pairs.tsv` generated
- [ ] `FINAL_MATCHES ⊆ CANDIDATES` verified
- [ ] Official validator PASS
- [ ] All tests PASS
- [ ] `README.md`, `docs/methodology.md`, `docs/experiments.md`, `docs/error_analysis.md` completed
- [ ] Final submission ZIP assembled

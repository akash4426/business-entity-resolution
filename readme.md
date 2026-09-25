# Business Entity Resolution — Final Architecture

## Core Objective

Resolve every Source 1 business against noisy Source 2 and Source 3
records while supporting:

- 0 matches
- 1 match
- Many matches

The system is optimized for macro F0.5, where precision is weighted
more heavily than recall.

---

                    ┌──────────────────────────┐
                    │       INPUT DATA         │
                    │                          │
                    │  Source 1   ~2.2M        │
                    │  Source 2   ~5.0M        │
                    │  Source 3   ~5.3M        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │        1. DATA QUALITY LAYER       │
              │                                    │
              │ • Schema validation                │
              │ • Missing-value analysis            │
              │ • Duplicate validation             │
              │ • Source consistency checks        │
              │ • Country distribution             │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       2. MULTI-VIEW NORMALIZER     │
              │                                    │
              │ Preserve ORIGINAL + create views   │
              │                                    │
              │ • Unicode NFKC                     │
              │ • Case normalization               │
              │ • Punctuation normalization       │
              │ • & ↔ and                         │
              │ • Legal suffix handling           │
              │ • Domain normalization             │
              │ • Accent normalization             │
              │ • Transliteration view             │
              │ • Address normalization            │
              │ • Tokenization                     │
              │ • Character n-grams                │
              │                                    │
              │ IMPORTANT: never destroy original  │
              │ representation                     │
              └──────────────────┬─────────────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  │                              │
                  ▼                              ▼
      ┌───────────────────────┐       ┌───────────────────────┐
      │ 3A. LEXICAL INDEXES   │       │ 3B. VECTOR INDEX      │
      │                       │       │                       │
      │ • Exact name          │       │ Multilingual          │
      │ • Token inverted      │       │ embeddings            │
      │ • Character n-gram    │       │                       │
      │ • Phonetic            │       │ Name embedding        │
      │ • Address tokens      │       │ Address embedding     │
      │ • Country             │       │ Combined embedding    │
      └───────────┬───────────┘       └───────────┬───────────┘
                  │                               │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │     4. HIGH-RECALL RETRIEVAL       │
              │                                    │
              │ Generate candidates independently  │
              │ from multiple channels             │
              │                                    │
              │ Exact normalized retrieval         │
              │ Token blocking                     │
              │ Character retrieval                │
              │ Phonetic retrieval                 │
              │ Address retrieval                  │
              │ Embedding ANN retrieval            │
              │                                    │
              │              UNION                 │
              │                ↓                   │
              │        Candidate Set              │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       5. CANDIDATE EVALUATION      │
              │                                    │
              │ Measure BEFORE MATCHING:           │
              │                                    │
              │ • Candidate Recall                │
              │ • Recall by country               │
              │ • Recall by difficulty             │
              │ • Candidate count                 │
              │ • Reduction ratio                 │
              │ • Runtime                         │
              │                                    │
              │ If true match isn't here →        │
              │ downstream model cannot recover it │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       6. PAIRWISE FEATURE         │
              │             ENGINE                 │
              │                                    │
              │ NAME FEATURES                      │
              │ • RapidFuzz ratios                 │
              │ • Jaro-Winkler                     │
              │ • Edit distance                    │
              │ • Token Jaccard                    │
              │ • Character similarity             │
              │                                    │
              │ ADDRESS FEATURES                   │
              │ • Token overlap                    │
              │ • Character similarity             │
              │ • Component similarity              │
              │ • Number similarity                │
              │ • Postal-code similarity            │
              │                                    │
              │ SEMANTIC FEATURES                  │
              │ • Name embedding similarity        │
              │ • Address embedding similarity     │
              │ • Combined similarity              │
              │                                    │
              │ STRUCTURAL FEATURES                │
              │ • Country agreement                │
              │ • Name length difference           │
              │ • Address length difference        │
              │ • Missing-address indicators       │
              │ • Source indicators                │
              │ • Retrieval-channel indicators     │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       7. PAIRWISE MATCHER         │
              │                                    │
              │ Baseline                          │
              │ Logistic Regression               │
              │                                    │
              │ Final classical model             │
              │ LightGBM / equivalent             │
              │                                    │
              │ Output: P(same_entity | pair)     │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │      8. PROBABILITY CALIBRATION   │
              │                                    │
              │ • Platt scaling / isotonic         │
              │ • Leakage-safe calibration         │
              │ • Reliability analysis             │
              │                                    │
              │ Convert raw score → probability   │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       9. DECISION ENGINE           │
              │                                    │
              │ For each Source 1 entity:          │
              │                                    │
              │ • Absolute threshold               │
              │ • Relative score                   │
              │ • Score margin                     │
              │ • Candidate density                │
              │ • Ambiguity detection              │
              │                                    │
              │ Output: 0 / 1 / MANY               │
              └──────────────────┬─────────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │ HIGH CONFIDENCE        │  │ AMBIGUOUS PAIRS        │
        │                        │  │                        │
        │ Accept / reject        │  │ Local LLM judge        │
        │ directly               │  │ ONLY here              │
        └────────────┬───────────┘  └────────────┬───────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │    10. GLOBAL CONSISTENCY LAYER    │
              │                                    │
              │ Training evidence shows:           │
              │ S2/S3 IDs map to only one S1       │
              │                                    │
              │ Therefore experimentally evaluate: │
              │                                    │
              │ • Ownership conflicts              │
              │ • Duplicate assignments            │
              │ • Score-based resolution            │
              │ • Global consistency               │
              │                                    │
              │ Enable only if validation improves │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │       11. FINAL VALIDATION         │
              │                                    │
              │ • Macro F0.5                      │
              │ • Precision                       │
              │ • Recall                          │
              │ • Candidate Recall                │
              │ • Singleton F0.5                  │
              │ • Multi-match F0.5                │
              │ • India performance               │
              │ • US performance                  │
              │ • Unseen France performance       │
              │ • Runtime                          │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │        12. EXPLAINABILITY          │
              │                                    │
              │ • Feature importance                │
              │ • SHAP                             │
              │ • Error analysis                   │
              │ • Ablation study                  │
              │ • Blocking contribution            │
              │ • LLM contribution                │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │             OUTPUT                 │
              │                                    │
              │ output/matching_results.tsv        │
              │ output/candidate_pairs.tsv         │
              └────────────────────────────────────┘

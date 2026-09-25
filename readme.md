# Business Entity Resolution

**Reality check first:** the full architecture in `ARCHITECTURE.md` is the _ideal_ system. In 72 hours with 3 people, you cannot build all of it well — trying to will leave you with nothing that runs. This plan tells you exactly what to cut, what to protect, and in what order, so you always have a submittable, valid output on hand.

---

## 0. Prioritization — what's Must / Should / Cut

| Tier                                                | Components                                                                                                                                                                                                                                                                    | Why                                                                                                                                                                                           |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST (protect at all costs)**                     | Data contract + validation, leakage-safe split, basic normalization, 2–3 retrieval channels (exact + token + one fuzzy/embedding), candidate recall check, pairwise features, LightGBM matcher, 0/1/many decision engine, threshold tuning, output generation, validator PASS | Without these you have no scoreable submission. This is the entire critical path.                                                                                                             |
| **SHOULD (add if on schedule)**                     | Phonetic + address retrieval channels, embedding ANN retrieval, hard negative mining, calibration, groupby-argmax global consistency, France-specific check, basic error analysis                                                                                             | These are what separate a decent F0.5 from a good one, and what make the methodology doc credible.                                                                                            |
| **CUT unless significant time remains at hour 60+** | MinHash LSH, Sorted Neighborhood, selective local LLM, full A0–A12 ablation suite, SHAP, extensive error-bucket taxonomy                                                                                                                                                      | Real, measurable value, but not worth the risk of missing the MUST tier. Mention them in the doc as "identified but deferred due to time constraints" — this is honest and still shows depth. |

**Golden rule for 72 hours: get a complete, valid, mediocre submission by hour 24. Improve it after that. Never let "make it good" block "make it exist."**

---

## 1. Team of 3 — Role Split

| Role                          | Owns                                                                                                                        | Focus                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **A — Data & Retrieval**      | Data contract, split, normalization, all retrieval channels, candidate recall gate, freeze                                  | Gets a valid, high-recall candidate set to B as fast as possible         |
| **B — Modeling**              | Feature engine, LightGBM, calibration, hard negatives                                                                       | Turns candidates into scored pairs                                       |
| **C — Decision, Eval & Docs** | Metric implementation, decision engine, threshold tuning, global consistency, output generation, validator, methodology doc | Turns scores into a valid submission, and writes the story as it happens |

With 3 people, everyone touches the pipeline sequentially at some point — the split above is about **primary ownership**, not isolation. Pair up at the two hard handoff points (see below).

---

## 2. Condensed Step Diagram (72hr-scoped)

```
[A] Data contract + validation ─┐
[A] Leakage-safe S1 split       ├─► hour 0–8
[A] Basic normalization (name+address, no phonetic yet)
              │
              v
[A] Retrieval: exact + token + character/fuzzy ─┐
[A] (if time) + embedding ANN (small model)      ├─► hour 8–20
              │
              v
   🚦 CANDIDATE RECALL CHECK (on val split)  ◄── hour ~20, WHOLE TEAM REVIEWS
   (target: recall good enough that misses aren't the bottleneck —
    don't chase perfection here, move on once "good enough")
              │
              v
[A→B handoff] FREEZE candidate generation ─────► hour ~22
              │
              v
[B] Pairwise features (name/address lexical + structural;
    embedding cosine only if retrieval used embeddings)     ─► hour 22–32
[B] Logistic regression baseline (quick sanity check)
[B] LightGBM matcher
[B] (if time) hard negatives + calibration
              │
              v
[B→C handoff] Scored candidate pairs ──────────► hour ~36
              │
              v
[C] Macro F0.5 metric (built earlier, in parallel — see below)
[C] 0/1/many decision engine + threshold tuning on val        ─► hour 36–48
[C] (if time) groupby-argmax global consistency check
              │
              v
   🚦 VALIDATION MACRO F0.5 LOCKED  ◄── hour ~48, WHOLE TEAM REVIEWS
              │
              v
[A+B+C] Run frozen pipeline on FULL test set                  ─► hour 48–58
[C] Generate matching_results.tsv + candidate_pairs.tsv
[C] Run utils/validate_submission.py → 🚦 MUST PASS            ─► hour ~58
              │
              v
[ALL] Buffer for bugs found during test-scale run              ─► hour 58–66
[C] Finish methodology.md, README, requirements.txt
[A] Package submission ZIP                                     ─► hour 66–72
              │
              v
          SUBMIT (with margin — do not submit at hour 71:59)
```

---

## 3. Hour-by-Hour Timeline

_(Assumes a team that sleeps in shifts, not a straight 72-hour grind — build in rest or you will make costly mistakes at hour 50.)_

| Hours     | A (Data & Retrieval)                                                                                                      | B (Modeling)                                                                       | C (Decision, Eval & Docs)                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **0–4**   | Inspect repo/data, build data contract + validation + tests                                                               | Set up env, dependencies, LightGBM/sklearn sanity install                          | Implement Macro F0.5 metric function + unit test it against the worked example in the problem statement |
| **4–8**   | Leakage-safe S1 split; start normalization (name first)                                                                   | Draft feature-engine skeleton against synthetic/dummy candidate pairs              | Draft decision-engine skeleton (thresholding logic) against synthetic scores                            |
| **8–14**  | Finish normalization (address); build exact + token retrieval                                                             | Continue feature engine (structural + lexical features), ready to run on real data | Build output-file writer + wrap `utils/validate_submission.py` into a quick local check script          |
| **14–20** | Build character/fuzzy retrieval channel; (stretch) small embedding ANN channel; union candidates                          | Idle/pairing with A to unblock retrieval, or start writing feature unit tests      | Start `docs/eda_report.md` and methodology skeleton with real EDA numbers                               |
| **~20**   | 🚦 **Candidate recall check on val split — all 3 review together**                                                        |                                                                                    |                                                                                                         |
| **20–22** | Freeze candidate generation config                                                                                        | —                                                                                  | —                                                                                                       |
| **22–28** | Support B if retrieval bugs surface; otherwise start France-specific normalization check                                  | Run real feature engine on frozen candidates; train LR baseline                    | Finalize decision-engine logic (0/1/many branches), write tests                                         |
| **28–36** | Idle/QA pass on retrieval code; write retrieval unit tests                                                                | Train LightGBM, compare vs. baseline; add calibration if time allows               | Wire decision engine to real (small/dev-scale) scored output; smoke-test end-to-end on val split        |
| **36–42** | —                                                                                                                         | Hand off scored candidates to C; support threshold tuning if needed                | Run threshold/margin optimization for Macro F0.5 on val                                                 |
| **42–48** | (stretch) groupby-argmax global consistency, if B/C have bandwidth to support                                             | (stretch) hard-negative mining pass if ahead of schedule                           | Evaluate global consistency impact; lock final decision config                                          |
| **~48**   | 🚦 **Validation Macro F0.5 locked — all 3 review together, decide what's in/out of final pipeline**                       |                                                                                    |                                                                                                         |
| **48–54** | Run frozen normalization + retrieval on full test set (this will take real wall-clock time — start early, monitor memory) | Run feature engine + matcher on full test candidates as A's output streams in      | Prep output-generation script for full-scale run                                                        |
| **54–58** | Support/debug full-scale run (this is where scale bugs appear — buffer time is critical)                                  | Same                                                                               | Generate `matching_results.tsv` + `candidate_pairs.tsv`; run validator                                  |
| **58–62** | Fix any validator failures found                                                                                          | Fix any validator failures found                                                   | Fix any validator failures found — **all hands if validator fails**                                     |
| **62–66** | Write README + reproducibility instructions for retrieval module                                                          | Write methodology sections for features/modeling                                   | Finish methodology.md, error analysis (lightweight), experiments summary                                |
| **66–70** | Assemble submission ZIP structure                                                                                         | Final code cleanup, pin `requirements.txt`                                         | Final doc pass, proofread                                                                               |
| **70–72** | **Buffer / submit**                                                                                                       | **Buffer / submit**                                                                | **Buffer / submit**                                                                                     |

---

## 4. Hard Rules for a 72-Hour Sprint

1. **Subsample aggressively while developing.** Don't run anything against the full multi-million-row dataset until the hour-48+ full test run. Use a 20–50K row sample (stratified by country and match-count) for all iteration — this alone is the difference between 10 iterations and 1.
2. **Two hard sync points, not daily standups.** At 72 hours you don't have time for ceremony — sync hard at the candidate-freeze (~hour 20) and the Macro-F0.5-lock (~hour 48), and otherwise communicate asynchronously (shared doc/channel) so nobody blocks on a meeting.
3. **Have a valid submission by hour 24, even if it's just exact+token retrieval and a logistic regression matcher.** A mediocre, valid, on-time submission beats a sophisticated one that isn't finished. Improve iteratively from there.
4. **Whoever finishes their block early pairs with whoever's behind — don't let one person's slip become the team's slip.** With only 3 people there's no slack for silos.
5. **Freeze points are real freezes.** Once candidates are frozen (~hour 22), Retrieval (A) does not keep tweaking it "just a little more" — that invalidates B and C's work in progress. Same for the Macro F0.5 lock.
6. **Budget real wall-clock time for the full-scale test run** (hour 48–58) — at millions of rows, even an efficient pipeline can take hours per stage. Start it as early as your validation numbers justify, and treat any surprise slowdown as your top priority, not a side task.
7. **Write the methodology doc as you go**, not from hour 62. You will not remember your reasoning for a threshold choice made at hour 10 by hour 68.
8. **What you cut, name explicitly in the doc.** A line like _"MinHash LSH and selective LLM re-ranking were identified as valuable but deferred due to the 72-hour constraint"_ reads as engineering judgment, not as a gap — reviewers notice the difference between "didn't think of it" and "chose not to, and said why."

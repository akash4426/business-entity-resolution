import polars as pl
import pandas as pd
import json
import os
from collections import defaultdict
import numpy as np

def run_diagnostics():
    base_dir = "/Users/akashmacbook/Desktop/business-entity-resolution"
    artifacts_dir = f"{base_dir}/code/business_entity_resolution/artifacts"
    docs_dir = f"{base_dir}/docs"
    
    # LOAD DATA
    val_s1 = pl.read_parquet(f"{artifacts_dir}/processed/val_s1_split.parquet")
    val_gt = pl.read_csv(f"{artifacts_dir}/splits/val_gt_split.tsv", separator='\t').fill_null("")
    val_cands = pl.read_parquet(f"{artifacts_dir}/val_candidates.parquet")
    val_preds = pl.read_parquet(f"{artifacts_dir}/val_predictions.parquet")
    val_features = pl.read_parquet(f"{artifacts_dir}/val_features.parquet")
    val_results = pl.read_csv(f"{artifacts_dir}/val_matching_results.tsv", separator='\t').fill_null("")
    
    # Clean GT strings (we know they are comma-separated without spaces)
    # explode GT
    gt_exploded = val_gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # explode PREDS (final results)
    results_exploded = val_results.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # ----------------------------------------------------
    # CANDIDATE RECALL
    # ----------------------------------------------------
    total_val_s1 = val_s1.height
    
    # Count true links
    total_true_links = gt_exploded.height
    
    # True links in candidates
    cands_set = set(zip(val_cands["source1_entity_id"].to_list(), val_cands["candidate_entity_id"].to_list()))
    gt_list = list(zip(gt_exploded["source1_entity_id"].to_list(), gt_exploded["matched_entity_ids"].to_list()))
    
    recovered_links = sum(1 for link in gt_list if link in cands_set)
    pair_recall = recovered_links / total_true_links if total_true_links > 0 else 0
    
    # Entity recall
    # Map S1 -> set of true matches
    gt_map = defaultdict(set)
    for s1_id, match_id in gt_list:
        gt_map[s1_id].add(match_id)
        
    cand_map = defaultdict(set)
    for s1_id, cand_id in cands_set:
        cand_map[s1_id].add(cand_id)
        
    # We must consider ALL S1s, including those with 0 true matches.
    # An S1 with 0 true matches has ALL true matches in candidates (trivially).
    all_s1_ids = val_s1["entity_id"].to_list()
    
    all_recovered_count = 0
    at_least_one_count = 0
    
    for s1_id in all_s1_ids:
        true_matches = gt_map.get(s1_id, set())
        cand_matches = cand_map.get(s1_id, set())
        
        if len(true_matches) == 0:
            all_recovered_count += 1
            at_least_one_count += 1 # Trivially true
        else:
            intersection = true_matches.intersection(cand_matches)
            if len(intersection) == len(true_matches):
                all_recovered_count += 1
            if len(intersection) > 0:
                at_least_one_count += 1
                
    entity_recall = all_recovered_count / total_val_s1
    at_least_one_recall = at_least_one_count / total_val_s1
    
    # ----------------------------------------------------
    # CANDIDATE SIZE DISTRIBUTION
    # ----------------------------------------------------
    cand_counts = val_cands.group_by("source1_entity_id").len()
    # add missing S1s (0 candidates)
    s1_df = pl.DataFrame({"source1_entity_id": all_s1_ids})
    cand_counts_full = s1_df.join(cand_counts, on="source1_entity_id", how="left").fill_null(0)
    
    sizes = cand_counts_full["len"].to_numpy()
    cand_mean = float(np.mean(sizes))
    cand_median = float(np.median(sizes))
    cand_p90 = float(np.percentile(sizes, 90))
    cand_p95 = float(np.percentile(sizes, 95))
    cand_p99 = float(np.percentile(sizes, 99))
    cand_max = int(np.max(sizes))
    cand_min = int(np.min(sizes))
    
    def pct_in_range(low, high):
        if high is None:
            return float(np.sum(sizes >= low) / total_val_s1)
        return float(np.sum((sizes >= low) & (sizes <= high)) / total_val_s1)
        
    cand_dist = {
        "0": pct_in_range(0, 0),
        "1": pct_in_range(1, 1),
        "2-5": pct_in_range(2, 5),
        "6-10": pct_in_range(6, 10),
        "11-25": pct_in_range(11, 25),
        "26-50": pct_in_range(26, 50),
        "51-100": pct_in_range(51, 100),
        "101-500": pct_in_range(101, 500),
        "501-1000": pct_in_range(501, 1000),
        "1000+": pct_in_range(1001, None)
    }
    
    # ----------------------------------------------------
    # FINAL MODEL PERFORMANCE ON VALIDATION
    # ----------------------------------------------------
    pred_map = defaultdict(set)
    res_list = list(zip(results_exploded["source1_entity_id"].to_list(), results_exploded["matched_entity_ids"].to_list()))
    for s1_id, match_id in res_list:
        pred_map[s1_id].add(match_id)
        
    f05_scores = []
    precisions = []
    recalls = []
    
    tps = 0
    fps = 0
    fns = 0
    
    # To compute Cardinality metrics later
    actual_cardinality = defaultdict(int)
    predicted_cardinality = defaultdict(int)
    
    for s1_id in all_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        
        actual_card = len(true_set)
        pred_card = len(pred_set)
        
        actual_card_str = str(actual_card) if actual_card <= 5 else "6+"
        pred_card_str = str(pred_card) if pred_card <= 5 else "6+"
        
        actual_cardinality[actual_card_str] += 1
        predicted_cardinality[pred_card_str] += 1
        
        if not true_set and not pred_set:
            p, r, f = 1.0, 1.0, 1.0
        elif not true_set and pred_set:
            p, r, f = 0.0, 0.0, 0.0
            fps += len(pred_set)
        elif true_set and not pred_set:
            p, r, f = 0.0, 0.0, 0.0
            fns += len(true_set)
        else:
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            tps += tp
            fps += fp
            fns += fn
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
            
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f)
        
    macro_f05 = float(np.mean(f05_scores))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    
    micro_p = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    micro_r = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    
    # ----------------------------------------------------
    # RETRIEVAL VS MODEL VS DECISION ERROR DECOMPOSITION
    # ----------------------------------------------------
    # Error classification per S1:
    # A. RETRIEVAL FAILURE: At least one true match is absent from candidates.
    # B. MODEL FAILURE: True match in cands, but scores < 0.5 (FN) OR false match scores >= 0.5 (FP)
    # C. DECISION FAILURE: Not applicable here since we use direct 0.5 thresholding (no global consistency).
    # D. CORRECT: F0.5 == 1.0
    
    err_A = 0
    err_B = 0
    err_D = 0
    
    # For predictions, probability is in val_preds
    # Create dict mapping (s1, cand) -> prob
    prob_dict = {(r[0], r[1]): r[2] for r in val_preds.select(["source1_entity_id", "candidate_entity_id", "probability"]).iter_rows()}
    
    for s1_id in all_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        cand_set = cand_map.get(s1_id, set())
        
        if true_set == pred_set:
            err_D += 1
            continue
            
        # Is there a true match missing from candidates?
        missing_from_cands = true_set - cand_set
        if len(missing_from_cands) > 0:
            err_A += 1
        else:
            # All true matches in candidates, so it's a model/threshold failure
            err_B += 1
            
    pct_A = err_A / total_val_s1
    pct_B = err_B / total_val_s1
    pct_C = 0.0
    pct_D = err_D / total_val_s1
    
    # ----------------------------------------------------
    # TOP-K ANALYSIS
    # ----------------------------------------------------
    topk_res = {k: 0 for k in [1, 2, 3, 5, 10, 20, 50, 100]}
    
    # Group predictions by s1_id and sort by prob descending
    preds_sorted = val_preds.sort(["source1_entity_id", "probability"], descending=[False, True])
    # Extract to dict {s1: [cand1, cand2, ...]}
    s1_ranked_cands = defaultdict(list)
    for row in preds_sorted.select(["source1_entity_id", "candidate_entity_id"]).iter_rows():
        s1_ranked_cands[row[0]].append(row[1])
        
    topk_at_least_one = {k: 0 for k in [1, 2, 3, 5, 10, 20, 50, 100]}
    topk_all = {k: 0 for k in [1, 2, 3, 5, 10, 20, 50, 100]}
    
    # Only evaluate Top-K for S1s that have true matches
    s1_with_matches = [s1 for s1 in all_s1_ids if len(gt_map.get(s1, set())) > 0]
    total_s1_matches = len(s1_with_matches)
    
    for s1_id in s1_with_matches:
        true_set = gt_map.get(s1_id, set())
        ranked = s1_ranked_cands.get(s1_id, [])
        
        for k in topk_res.keys():
            top_k_set = set(ranked[:k])
            intersect = true_set.intersection(top_k_set)
            
            if len(intersect) > 0:
                topk_at_least_one[k] += 1
            if len(intersect) == len(true_set):
                topk_all[k] += 1
                
    for k in topk_res.keys():
        if total_s1_matches > 0:
            topk_at_least_one[k] = topk_at_least_one[k] / total_s1_matches
            topk_all[k] = topk_all[k] / total_s1_matches
            
    # ----------------------------------------------------
    # S1->S2 vs S1->S3
    # ----------------------------------------------------
    s2_gt = sum(1 for _, m in gt_list if str(m).startswith("S2-"))
    s3_gt = sum(1 for _, m in gt_list if str(m).startswith("S3-"))
    
    s2_cands = sum(1 for _, c in cands_set if str(c).startswith("S2-"))
    s3_cands = sum(1 for _, c in cands_set if str(c).startswith("S3-"))
    
    s2_recovered = sum(1 for _, m in gt_list if str(m).startswith("S2-") and (s1_id, m) in cands_set)
    s3_recovered = sum(1 for _, m in gt_list if str(m).startswith("S3-") and (s1_id, m) in cands_set)
    
    # ----------------------------------------------------
    # WRITE REPORT
    # ----------------------------------------------------
    metrics = {
        "validation_s1_count": total_val_s1,
        "candidate_pair_recall": pair_recall,
        "candidate_entity_recall": entity_recall,
        "at_least_one_candidate_recall": at_least_one_recall,
        "cand_mean": cand_mean,
        "cand_p95": cand_p95,
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "retrieval_failure_pct": pct_A,
        "model_failure_pct": pct_B,
        "decision_failure_pct": pct_C,
        "top10_recall_all": topk_all[10],
        "s2_gt_links": s2_gt,
        "s3_gt_links": s3_gt
    }
    
    with open(f"{artifacts_dir}/first_submission_diagnostics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    bottleneck = "RETRIEVAL" if pct_A > pct_B else "MODELING"
    
    report = f"""# First Submission Diagnostics

## Current Submission
The pipeline uses exact name and exact address blocking channels, yielding ~0.609 validation Macro F0.5.

## Validation Dataset
- Total S1 Entities: {total_val_s1}
- Total Ground Truth Links: {total_true_links}

## Candidate Recall
- Pair Recall (Links): {pair_recall:.4f}
- Entity Recall (All true matches present): {entity_recall:.4f}
- At-least-one Recall: {at_least_one_recall:.4f}

## Candidate Size
- Mean: {cand_mean:.2f}
- Median: {cand_median:.2f}
- P90: {cand_p90:.2f}
- P95: {cand_p95:.2f}
- Max: {cand_max}

## Model Performance
- Macro F0.5: {macro_f05:.4f}
- Macro Precision: {macro_p:.4f}
- Macro Recall: {macro_r:.4f}
- Total TP: {tps}, Total FP: {fps}, Total FN: {fns}

## Actual vs Predicted Cardinality
| Match Count | Actual | Predicted |
|-------------|--------|-----------|
| 0 | {actual_cardinality['0']} | {predicted_cardinality['0']} |
| 1 | {actual_cardinality['1']} | {predicted_cardinality['1']} |
| 2 | {actual_cardinality['2']} | {predicted_cardinality['2']} |
| 3 | {actual_cardinality['3']} | {predicted_cardinality['3']} |
| 4 | {actual_cardinality['4']} | {predicted_cardinality['4']} |
| 5 | {actual_cardinality['5']} | {predicted_cardinality['5']} |
| 6+ | {actual_cardinality['6+']} | {predicted_cardinality['6+']} |

## Retrieval vs Model vs Decision Errors
- **Retrieval Failure**: {pct_A:.2%} (True match missed by candidates)
- **Model Failure**: {pct_B:.2%} (True match in candidates but incorrectly scored)
- **Decision Failure**: {pct_C:.2%} (Thresholding issues)
- **Correct**: {pct_D:.2%}

## Top-K Recall (All True Matches)
- Top 1: {topk_all[1]:.4f}
- Top 5: {topk_all[5]:.4f}
- Top 10: {topk_all[10]:.4f}
- Top 50: {topk_all[50]:.4f}

## Main Bottleneck
**{bottleneck}** is the primary bottleneck.

## Recommended Next Experiment
Focus on improving the {bottleneck} phase.
"""
    
    with open(f"{docs_dir}/first_submission_diagnostics.md", "w") as f:
        f.write(report)
        
    print(f"Diagnostics complete. F0.5={macro_f05:.4f}")

if __name__ == "__main__":
    run_diagnostics()

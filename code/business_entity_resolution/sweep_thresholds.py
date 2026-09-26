import polars as pl
from pathlib import Path
import json

gt_path = "artifacts/splits/val_gt_v2_split.tsv"
preds_path = "artifacts/val_predictions_v2.parquet"

gt = pl.read_csv(gt_path, separator='\t')
preds_all = pl.read_parquet(preds_path)

# we need to make gt a dict of sets for fast eval
import pandas as pd
gt_pd = gt.to_pandas()
gt_dict = {}
for _, row in gt_pd.iterrows():
    s1 = row["source1_entity_id"]
    matches = row["matched_entity_ids"]
    if pd.isna(matches) or matches == "":
        gt_dict[s1] = set()
    else:
        gt_dict[s1] = set(matches.split(","))

def eval_threshold(threshold):
    matches = preds_all.filter(pl.col("probability") >= threshold)
    matches_grouped = matches.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("matches")
    ).with_columns(
        pl.col("matches").list.join(",")
    )
    s1 = pl.read_parquet("artifacts/processed_v2/val_s1_v2_split.parquet")
    all_s1 = s1.select(["entity_id"]).rename({"entity_id": "source1_entity_id"})
    
    final_matches = all_s1.join(matches_grouped, on="source1_entity_id", how="left").fill_null("")
    
    fm_pd = final_matches.to_pandas()
    pred_dict = {}
    for _, row in fm_pd.iterrows():
        s1_id = row["source1_entity_id"]
        m = row["matches"]
        if pd.isna(m) or m == "":
            pred_dict[s1_id] = set()
        else:
            pred_dict[s1_id] = set(m.split(","))
            
    f05_scores = []
    precisions = []
    recalls = []
    for s1_id, y_true in gt_dict.items():
        y_pred = pred_dict.get(s1_id, set())
        
        if len(y_true) == 0 and len(y_pred) == 0:
            f05_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        elif len(y_true) == 0 and len(y_pred) > 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(1.0)
            continue
        elif len(y_true) > 0 and len(y_pred) == 0:
            f05_scores.append(0.0)
            precisions.append(1.0)
            recalls.append(0.0)
            continue
            
        tp = len(y_true.intersection(y_pred))
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            
        f05_scores.append(f05)
        precisions.append(precision)
        recalls.append(recall)
        
    return sum(f05_scores) / len(f05_scores), sum(precisions) / len(precisions), sum(recalls) / len(recalls)

for t in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    f05, p, r = eval_threshold(t)
    print(f"Threshold: {t:.2f} | Macro F0.5: {f05:.4f} | P: {p:.4f} | R: {r:.4f}")

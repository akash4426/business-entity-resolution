import pandas as pd
import argparse
import json

def calculate_f05(gt_path, pred_path, metrics_path):
    print("Loading Ground Truth and Predictions for evaluation...")
    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    pred = pd.read_csv(pred_path, sep='\t', dtype=str).fillna('')
    
    # Must have same S1 IDs
    assert len(gt) == len(pred), "Row counts do not match!"
    
    merged = pd.merge(gt, pred, on='source1_entity_id')
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for _, row in merged.iterrows():
        gt_set = set([x.strip() for x in row['matched_entity_ids_x'].split(',')] if row['matched_entity_ids_x'].strip() else [])
        pred_set = set([x.strip() for x in row['matched_entity_ids_y'].split(',')] if row['matched_entity_ids_y'].strip() else [])
        
        # If both are empty, it's a true negative for this S1, which means perfect precision/recall for this S1.
        # But macro F0.5 is over all S1 entities. Wait, the problem says:
        # "The challenge metric is macro-average F0.5 over Source 1 entities."
        # This implies we calculate F0.5 per S1 entity and average them.
        pass
        
    # Let's properly implement macro-average F0.5 over Source 1 entities
    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    for _, row in merged.iterrows():
        gt_set = set([x.strip() for x in row['matched_entity_ids_x'].split(',')] if row['matched_entity_ids_x'].strip() else [])
        pred_set = set([x.strip() for x in row['matched_entity_ids_y'].split(',')] if row['matched_entity_ids_y'].strip() else [])
        
        if not gt_set and not pred_set:
            # True empty + predicted empty
            p, r, f = 1.0, 1.0, 1.0
        elif not gt_set and pred_set:
            p, r, f = 0.0, 0.0, 0.0
        elif gt_set and not pred_set:
            p, r, f = 0.0, 0.0, 0.0
        else:
            tp = len(gt_set.intersection(pred_set))
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if p + r == 0:
                f = 0.0
            else:
                f = (1.25 * p * r) / (0.25 * p + r)
                
        precision_scores.append(p)
        recall_scores.append(r)
        f05_scores.append(f)
        
    macro_p = sum(precision_scores) / len(precision_scores)
    macro_r = sum(recall_scores) / len(recall_scores)
    macro_f05 = sum(f05_scores) / len(f05_scores)
    
    metrics = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05
    }
    
    print(f"Validation Macro Precision: {macro_p:.4f}")
    print(f"Validation Macro Recall: {macro_r:.4f}")
    print(f"Validation Macro F0.5: {macro_f05:.4f}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()
    
    calculate_f05(args.gt, args.pred, args.metrics)

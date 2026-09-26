import pandas as pd
import json
import numpy as np
import argparse
from pathlib import Path

def generate_split(train_s1_path, gt_path, artifacts_dir, seed=42, val_size=0.2):
    print("Loading train S1 and ground truth for splitting...")
    train_s1 = pd.read_csv(train_s1_path, sep='\t', dtype=str, keep_default_na=False).fillna('')
    gt = pd.read_csv(gt_path, sep='\t', dtype=str, keep_default_na=False).fillna('')
    
    # We split based on S1 entity_ids
    # We want to stratify by country if possible to keep distributions similar
    
    np.random.seed(seed)
    # Shuffle
    train_s1_shuffled = train_s1.sample(frac=1.0, random_state=seed)
    
    # Calculate sizes
    total_size = len(train_s1_shuffled)
    val_count = int(total_size * val_size)
    
    val_s1 = train_s1_shuffled.iloc[:val_count]
    train_split_s1 = train_s1_shuffled.iloc[val_count:]
    
    train_s1_ids = set(train_split_s1['entity_id'])
    val_s1_ids = set(val_s1['entity_id'])
    
    assert len(train_s1_ids.intersection(val_s1_ids)) == 0, "Leakage detected between train and val S1 IDs!"
    
    # Ground truth mapping
    gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: [m.strip() for m in str(x).split(',')] if str(x).strip() else [])
    gt['num_matches'] = gt['match_list'].apply(len)
    
    train_gt = gt[gt['source1_entity_id'].isin(train_s1_ids)]
    val_gt = gt[gt['source1_entity_id'].isin(val_s1_ids)]
    
    # Save the splits to files to avoid recalculating
    split_dir = artifacts_dir / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    
    train_split_s1.to_csv(split_dir / 'train_s1_split.tsv', sep='\t', index=False)
    val_s1.to_csv(split_dir / 'val_s1_split.tsv', sep='\t', index=False)
    
    train_gt.drop(columns=['match_list', 'num_matches']).to_csv(split_dir / 'train_gt_split.tsv', sep='\t', index=False)
    val_gt.drop(columns=['match_list', 'num_matches']).to_csv(split_dir / 'val_gt_split.tsv', sep='\t', index=False)
    
    # Generate manifest
    manifest = {
        "seed": seed,
        "train_count": len(train_split_s1),
        "validation_count": len(val_s1),
        "train_country_distribution": train_split_s1['country'].value_counts().to_dict(),
        "val_country_distribution": val_s1['country'].value_counts().to_dict(),
        "train_zero_match": int((train_gt['num_matches'] == 0).sum()),
        "train_singleton_match": int((train_gt['num_matches'] == 1).sum()),
        "train_multi_match": int((train_gt['num_matches'] > 1).sum()),
        "val_zero_match": int((val_gt['num_matches'] == 0).sum()),
        "val_singleton_match": int((val_gt['num_matches'] == 1).sum()),
        "val_multi_match": int((val_gt['num_matches'] > 1).sum()),
    }
    
    manifest_path = artifacts_dir / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Split completed successfully. Manifest saved to {manifest_path}")
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_s1', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--artifacts_dir', required=True)
    args = parser.parse_args()
    
    generate_split(args.train_s1, args.gt, Path(args.artifacts_dir))

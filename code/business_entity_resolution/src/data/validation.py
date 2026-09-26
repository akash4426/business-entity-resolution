import pandas as pd
from typing import List, Dict, Any, Tuple
from .schema import SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS

def check_empty(s: pd.Series) -> pd.Series:
    """Return a boolean series where True indicates the value is empty or whitespace-only."""
    return s.isna() | (s.astype(str).str.strip() == '') | (s.astype(str).str.lower() == 'nan') | (s.astype(str).str.lower() == 'none')

def validate_source(df: pd.DataFrame, source_prefix: str, name: str) -> Dict[str, Any]:
    """Validate a source dataset (S1, S2, or S3)."""
    errors = []
    
    # 1. Schema check
    missing_cols = [col for col in SOURCE_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # If entity_id is missing, we can't do ID checks
    if 'entity_id' not in df.columns:
        return {"status": "FAIL", "errors": errors}
        
    total_rows = len(df)
    
    # 2. Missingness
    missing_entity_id = int(check_empty(df['entity_id']).sum())
    missing_name = int(check_empty(df['business_name']).sum()) if 'business_name' in df.columns else total_rows
    missing_address = int(check_empty(df['business_address']).sum()) if 'business_address' in df.columns else total_rows
    missing_country = int(check_empty(df['country']).sum()) if 'country' in df.columns else total_rows
    
    if missing_entity_id > 0:
        errors.append(f"Found {missing_entity_id} missing/empty entity_id values.")
        
    # 3. Uniqueness
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df['entity_id'].duplicated().sum())
    if duplicate_ids > 0:
        errors.append(f"Found {duplicate_ids} duplicate entity_id values.")
        
    # 4. Prefix check
    invalid_prefix_mask = ~df['entity_id'].fillna('').astype(str).str.startswith(source_prefix) & ~check_empty(df['entity_id'])
    invalid_prefixes = int(invalid_prefix_mask.sum())
    if invalid_prefixes > 0:
        errors.append(f"Found {invalid_prefixes} IDs not starting with prefix {source_prefix}.")
        
    # 5. Countries
    country_counts = {}
    if 'country' in df.columns:
        valid_countries = df['country'][~check_empty(df['country'])]
        country_counts = valid_countries.value_counts().to_dict()
    
    status = "PASS" if not errors else "FAIL"
    
    return {
        "status": status,
        "name": name,
        "rows": total_rows,
        "columns": len(df.columns),
        "unique_entity_ids": int(df['entity_id'].nunique()),
        "missingness": {
            "entity_id": missing_entity_id,
            "business_name": missing_name,
            "business_address": missing_address,
            "country": missing_country
        },
        "duplicate_rows": duplicate_rows,
        "countries": country_counts,
        "errors": errors
    }

def validate_ground_truth(gt_df: pd.DataFrame, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate ground truth dataset against the loaded source dataframes."""
    errors = []
    
    # 1. Schema check
    missing_cols = [col for col in GROUND_TRUTH_COLUMNS if col not in gt_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        
    if 'source1_entity_id' not in gt_df.columns or 'matched_entity_ids' not in gt_df.columns:
        return {"status": "FAIL", "errors": errors}
        
    total_rows = len(gt_df)
    
    # 2. Missingness and Uniqueness
    missing_s1_id = int(check_empty(gt_df['source1_entity_id']).sum())
    if missing_s1_id > 0:
        errors.append(f"Found {missing_s1_id} missing/empty source1_entity_id values.")
        
    duplicate_s1 = int(gt_df['source1_entity_id'].duplicated().sum())
    if duplicate_s1 > 0:
        errors.append(f"Found {duplicate_s1} duplicate source1_entity_id values.")
        
    # 3. Referencing S1
    if 'entity_id' in s1_df.columns:
        valid_s1_ids = set(s1_df['entity_id'][~check_empty(s1_df['entity_id'])])
        gt_s1_ids = set(gt_df['source1_entity_id'][~check_empty(gt_df['source1_entity_id'])])
        invalid_s1_refs = int(len(gt_s1_ids - valid_s1_ids))
        if invalid_s1_refs > 0:
            errors.append(f"Found {invalid_s1_refs} source1_entity_id values in ground truth that do not exist in Source 1.")
            
    # 4. Parsing matched_entity_ids
    empty_matches = int(check_empty(gt_df['matched_entity_ids']).sum())
    non_empty_matches = total_rows - empty_matches
    
    total_s2_refs = 0
    total_s3_refs = 0
    invalid_match_refs = 0
    duplicate_matches_in_row = 0
    
    valid_s2_ids = set(s2_df['entity_id'][~check_empty(s2_df['entity_id'])]) if 'entity_id' in s2_df.columns else set()
    valid_s3_ids = set(s3_df['entity_id'][~check_empty(s3_df['entity_id'])]) if 'entity_id' in s3_df.columns else set()
    
    # Iterate over matches to count and validate
    for idx, row in gt_df.iterrows():
        matches_str = row['matched_entity_ids']
        if pd.isna(matches_str) or str(matches_str).strip() == '':
            continue
            
        matches_list = [m.strip() for m in str(matches_str).split(',') if m.strip()]
        
        # Check for duplicates in row
        if len(matches_list) != len(set(matches_list)):
            duplicate_matches_in_row += 1
            
        for match_id in matches_list:
            if match_id.startswith('S2-'):
                total_s2_refs += 1
                if match_id not in valid_s2_ids:
                    invalid_match_refs += 1
            elif match_id.startswith('S3-'):
                total_s3_refs += 1
                if match_id not in valid_s3_ids:
                    invalid_match_refs += 1
            else:
                invalid_match_refs += 1 # Not S2 or S3
                
    if duplicate_matches_in_row > 0:
        errors.append(f"Found {duplicate_matches_in_row} rows with duplicate matched IDs in the list.")
    if invalid_match_refs > 0:
        errors.append(f"Found {invalid_match_refs} invalid matched entity IDs (wrong prefix or does not exist in S2/S3).")
        
    status = "PASS" if not errors else "FAIL"
    
    return {
        "status": status,
        "rows": total_rows,
        "unique_s1_ids": int(gt_df['source1_entity_id'].nunique()),
        "empty_match_rows": empty_matches,
        "non_empty_match_rows": non_empty_matches,
        "total_referenced_s2_ids": total_s2_refs,
        "total_referenced_s3_ids": total_s3_refs,
        "invalid_references": invalid_match_refs,
        "duplicate_references_in_row": duplicate_matches_in_row,
        "errors": errors
    }

def validate_source_separation(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, Any]:
    """Explicitly verify that source IDs do not overlap."""
    errors = []
    
    s1_ids = set(s1_df['entity_id'][~check_empty(s1_df['entity_id'])]) if 'entity_id' in s1_df.columns else set()
    s2_ids = set(s2_df['entity_id'][~check_empty(s2_df['entity_id'])]) if 'entity_id' in s2_df.columns else set()
    s3_ids = set(s3_df['entity_id'][~check_empty(s3_df['entity_id'])]) if 'entity_id' in s3_df.columns else set()
    
    s1_s2_overlap = len(s1_ids.intersection(s2_ids))
    s1_s3_overlap = len(s1_ids.intersection(s3_ids))
    s2_s3_overlap = len(s2_ids.intersection(s3_ids))
    
    if s1_s2_overlap > 0:
        errors.append(f"Found {s1_s2_overlap} IDs overlapping between Source 1 and Source 2.")
    if s1_s3_overlap > 0:
        errors.append(f"Found {s1_s3_overlap} IDs overlapping between Source 1 and Source 3.")
    if s2_s3_overlap > 0:
        errors.append(f"Found {s2_s3_overlap} IDs overlapping between Source 2 and Source 3.")
        
    status = "PASS" if not errors else "FAIL"
    
    return {
        "status": status,
        "s1_s2_overlap": s1_s2_overlap,
        "s1_s3_overlap": s1_s3_overlap,
        "s2_s3_overlap": s2_s3_overlap,
        "errors": errors
    }

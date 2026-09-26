import pandas as pd
import pytest
from src.data.validation import validate_source, validate_ground_truth, validate_source_separation

def test_valid_source():
    df = pd.DataFrame({
        'entity_id': ['S1-1', 'S1-2'],
        'business_name': ['A', 'B'],
        'business_address': ['Addr A', 'Addr B'],
        'country': ['US', 'India']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'PASS'
    assert res['errors'] == []
    assert res['missingness']['entity_id'] == 0

def test_missing_required_column():
    df = pd.DataFrame({
        'entity_id': ['S1-1'],
        'business_name': ['A']
        # missing address and country
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'FAIL'
    assert any("Missing required columns" in e for e in res['errors'])

def test_null_entity_id():
    df = pd.DataFrame({
        'entity_id': ['S1-1', '', '  ', None],
        'business_name': ['A', 'B', 'C', 'D'],
        'business_address': ['A', 'B', 'C', 'D'],
        'country': ['US', 'US', 'US', 'US']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'FAIL'
    assert res['missingness']['entity_id'] == 3

def test_duplicate_entity_id():
    df = pd.DataFrame({
        'entity_id': ['S1-1', 'S1-1'],
        'business_name': ['A', 'B'],
        'business_address': ['A', 'B'],
        'country': ['US', 'US']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'FAIL'
    assert any("duplicate entity_id" in e for e in res['errors'])

def test_incorrect_source_prefix():
    df = pd.DataFrame({
        'entity_id': ['S1-1', 'S2-2'],
        'business_name': ['A', 'B'],
        'business_address': ['A', 'B'],
        'country': ['US', 'US']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'FAIL'
    assert any("prefix S1-" in e for e in res['errors'])

def test_overlapping_source_ids():
    s1 = pd.DataFrame({'entity_id': ['S1-1', 'S1-2']})
    s2 = pd.DataFrame({'entity_id': ['S2-1', 'S1-2']}) # overlap S1-2
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    res = validate_source_separation(s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert res['s1_s2_overlap'] == 1

def test_valid_ground_truth():
    s1 = pd.DataFrame({'entity_id': ['S1-1', 'S1-2', 'S1-3']})
    s2 = pd.DataFrame({'entity_id': ['S2-1', 'S2-2']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2'],
        'matched_entity_ids': ['S2-1, S3-1', '']
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'PASS'
    assert res['empty_match_rows'] == 1
    assert res['non_empty_match_rows'] == 1
    assert res['total_referenced_s2_ids'] == 1
    assert res['total_referenced_s3_ids'] == 1

def test_missing_s1_ground_truth():
    s1 = pd.DataFrame({'entity_id': ['S1-1']})
    s2 = pd.DataFrame({'entity_id': ['S2-1']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    gt = pd.DataFrame({
        'source1_entity_id': [''],
        'matched_entity_ids': ['S2-1']
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert any("missing/empty source1_entity_id" in e for e in res['errors'])

def test_duplicate_ground_truth_s1():
    s1 = pd.DataFrame({'entity_id': ['S1-1']})
    s2 = pd.DataFrame({'entity_id': ['S2-1']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-1'],
        'matched_entity_ids': ['S2-1', 'S3-1']
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert any("duplicate source1_entity_id" in e for e in res['errors'])

def test_invalid_s2_s3_reference():
    s1 = pd.DataFrame({'entity_id': ['S1-1']})
    s2 = pd.DataFrame({'entity_id': ['S2-1']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1'],
        'matched_entity_ids': ['S2-2, S4-1'] # S2-2 not in S2, S4-1 invalid prefix
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert res['invalid_references'] == 2

def test_duplicate_matched_id():
    s1 = pd.DataFrame({'entity_id': ['S1-1']})
    s2 = pd.DataFrame({'entity_id': ['S2-1']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1'],
        'matched_entity_ids': ['S2-1, S2-1']
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert res['duplicate_references_in_row'] == 1

def test_s1_id_used_as_match():
    s1 = pd.DataFrame({'entity_id': ['S1-1', 'S1-2']})
    s2 = pd.DataFrame({'entity_id': ['S2-1']})
    s3 = pd.DataFrame({'entity_id': ['S3-1']})
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1'],
        'matched_entity_ids': ['S2-1, S1-2']
    })
    res = validate_ground_truth(gt, s1, s2, s3)
    assert res['status'] == 'FAIL'
    assert res['invalid_references'] == 1

def test_missing_business_name_address():
    df = pd.DataFrame({
        'entity_id': ['S1-1'],
        'business_name': [''],
        'business_address': [None],
        'country': ['US']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'PASS' # missing name/addr is not fatal, just reported
    assert res['missingness']['business_name'] == 1
    assert res['missingness']['business_address'] == 1

def test_dynamic_country_discovery():
    df = pd.DataFrame({
        'entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'business_name': ['A', 'B', 'C'],
        'business_address': ['A', 'B', 'C'],
        'country': ['US', 'France', 'US']
    })
    res = validate_source(df, 'S1-', 'train_s1')
    assert res['status'] == 'PASS'
    assert res['countries'] == {'US': 2, 'France': 1}

import json
import os
import sys
from pathlib import Path

# Add src to python path so we can import data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_source, load_ground_truth
from src.data.validation import validate_source, validate_ground_truth, validate_source_separation

def main():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = project_root / 'dataset' / 'train'
    test_dir = project_root / 'dataset' / 'test'
    artifacts_dir = project_root / 'code' / 'business_entity_resolution' / 'artifacts'
    docs_dir = project_root / 'docs'
    
    print("Loading datasets...")
    train_s1 = load_source(str(train_dir / 'train_source1.tsv'))
    train_s2 = load_source(str(train_dir / 'train_source2.tsv'))
    train_s3 = load_source(str(train_dir / 'train_source3.tsv'))
    gt = load_ground_truth(str(train_dir / 'train_ground_truth.tsv'))
    
    test_s1 = load_source(str(test_dir / 'test_source1.tsv'))
    test_s2 = load_source(str(test_dir / 'test_source2.tsv'))
    test_s3 = load_source(str(test_dir / 'test_source3.tsv'))
    
    print("Validating sources...")
    train_s1_res = validate_source(train_s1, 'S1-', 'train_source1')
    train_s2_res = validate_source(train_s2, 'S2-', 'train_source2')
    train_s3_res = validate_source(train_s3, 'S3-', 'train_source3')
    
    test_s1_res = validate_source(test_s1, 'S1-', 'test_source1')
    test_s2_res = validate_source(test_s2, 'S2-', 'test_source2')
    test_s3_res = validate_source(test_s3, 'S3-', 'test_source3')
    
    print("Validating source separation...")
    train_sep_res = validate_source_separation(train_s1, train_s2, train_s3)
    test_sep_res = validate_source_separation(test_s1, test_s2, test_s3)
    
    print("Validating ground truth...")
    gt_res = validate_ground_truth(gt, train_s1, train_s2, train_s3)
    
    # Compile countries
    all_countries = {}
    for res in [train_s1_res, train_s2_res, train_s3_res]:
        for c, count in res.get('countries', {}).items():
            if c not in all_countries:
                all_countries[c] = {'train': 0, 'test': 0}
            all_countries[c]['train'] += count
            
    for res in [test_s1_res, test_s2_res, test_s3_res]:
        for c, count in res.get('countries', {}).items():
            if c not in all_countries:
                all_countries[c] = {'train': 0, 'test': 0}
            all_countries[c]['test'] += count
            
    # Check overall status
    statuses = [
        train_s1_res['status'], train_s2_res['status'], train_s3_res['status'],
        test_s1_res['status'], test_s2_res['status'], test_s3_res['status'],
        train_sep_res['status'], test_sep_res['status'], gt_res['status']
    ]
    overall_status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    
    # Generate data_contract.json
    contract = {
        "status": overall_status,
        "train": {
            "source1": train_s1_res,
            "source2": train_s2_res,
            "source3": train_s3_res,
            "ground_truth": gt_res,
            "source_separation": train_sep_res
        },
        "test": {
            "source1": test_s1_res,
            "source2": test_s2_res,
            "source3": test_s3_res,
            "source_separation": test_sep_res
        },
        "countries": all_countries,
        "validation_checks": [
            "Source schema validation",
            "Entity ID presence, uniqueness, and prefix validation",
            "Source separation (no ID overlap between S1/S2/S3)",
            "Ground truth schema validation",
            "Ground truth S1 ID existence in S1 dataset",
            "Ground truth matched IDs prefix and existence in S2/S3 datasets",
            "Ground truth empty matches allowed",
            "Ground truth duplicate matched IDs detection",
            "Missingness calculation for all sources",
            "Dynamic country discovery"
        ]
    }
    
    contract_path = artifacts_dir / 'data_contract.json'
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"Generated {contract_path}")
    
    # Generate data_validation_report.md
    report = []
    report.append("# Data Validation Report")
    report.append("\n## 1. Purpose")
    report.append("Validate the integrity, schema, source separation, and ground truth mapping of the Business Entity Resolution dataset.")
    
    report.append("\n## 2. Dataset Inventory & Missingness Summary")
    
    def format_source_table(res):
        return f"| {res['name']} | {res['rows']:,} | {res['unique_entity_ids']:,} | {res['missingness']['entity_id']:,} | {res['missingness']['business_name']:,} | {res['missingness']['business_address']:,} | {res['missingness']['country']:,} | {res['status']} |"
        
    report.append("| Dataset | Total Rows | Unique IDs | Missing IDs | Missing Names | Missing Addresses | Missing Countries | Status |")
    report.append("|---|---|---|---|---|---|---|---|")
    report.append(format_source_table(train_s1_res))
    report.append(format_source_table(train_s2_res))
    report.append(format_source_table(train_s3_res))
    report.append(format_source_table(test_s1_res))
    report.append(format_source_table(test_s2_res))
    report.append(format_source_table(test_s3_res))
    
    report.append("\n## 3. Schema & ID Validation")
    for res in [train_s1_res, train_s2_res, train_s3_res, test_s1_res, test_s2_res, test_s3_res]:
        if res['errors']:
            report.append(f"**{res['name']}** FAILED:")
            for e in res['errors']:
                report.append(f"- {e}")
        else:
            report.append(f"**{res['name']}** PASSED schema and ID checks.")
            
    report.append("\n## 4. Source Separation Validation")
    report.append(f"**Train source separation**: {train_sep_res['status']}")
    for e in train_sep_res['errors']:
        report.append(f"- {e}")
    report.append(f"**Test source separation**: {test_sep_res['status']}")
    for e in test_sep_res['errors']:
        report.append(f"- {e}")
        
    report.append("\n## 5. Ground-Truth Validation")
    report.append(f"**Status**: {gt_res['status']}")
    report.append(f"- Total Rows: {gt_res['rows']:,}")
    report.append(f"- Unique S1 IDs: {gt_res['unique_s1_ids']:,}")
    report.append(f"- Empty-match Rows: {gt_res['empty_match_rows']:,}")
    report.append(f"- Non-empty-match Rows: {gt_res['non_empty_match_rows']:,}")
    report.append(f"- Total Referenced S2 IDs: {gt_res['total_referenced_s2_ids']:,}")
    report.append(f"- Total Referenced S3 IDs: {gt_res['total_referenced_s3_ids']:,}")
    report.append(f"- Invalid References: {gt_res['invalid_references']:,}")
    report.append(f"- Duplicate References In Row: {gt_res.get('duplicate_references_in_row', gt_res.get('duplicate_references', 0)):,}")
    
    if gt_res['errors']:
        report.append("\n**Errors**:")
        for e in gt_res['errors']:
            report.append(f"- {e}")
            
    report.append("\n## 6. Country Discovery")
    report.append("| Country | Train Count | Test Count |")
    report.append("|---|---|---|")
    for c, counts in all_countries.items():
        report.append(f"| {c} | {counts['train']:,} | {counts['test']:,} |")
        
    report.append("\n## 7. Validation Checks Implemented")
    for c in contract['validation_checks']:
        report.append(f"- [x] {c}")
        
    report.append("\n## 8. Final Status")
    report.append(f"**Overall Status: {overall_status}**")
    
    report.append("\n## 9. Reproduction Command")
    report.append("```bash\ncd code/business_entity_resolution && PYTHONPATH=. python src/run_validation.py\n```")
    
    report_path = docs_dir / 'data_validation_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Generated {report_path}")

if __name__ == "__main__":
    main()

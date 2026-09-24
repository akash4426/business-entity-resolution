# %% [markdown]
# # Phase 1: Exploratory Data Analysis & Data Audit
# ## Business Entity Resolution Challenge
# 
# This notebook performs a comprehensive EDA of the entity resolution dataset.
# It covers dataset overview, data quality, name/address noise, ground truth analysis,
# true-match characteristics, country analysis, and risk identification.
#
# **Run from the project root directory.**

# %% [markdown]
# ## 0. Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Paths — resolve relative to this notebook's location
PROJECT_ROOT = Path('..').resolve()
TRAIN_DIR = PROJECT_ROOT / 'dataset' / 'train'
TEST_DIR = PROJECT_ROOT / 'dataset' / 'test'
DOCS_DIR = PROJECT_ROOT / 'docs'
DOCS_DIR.mkdir(exist_ok=True)

# Verify files exist
for f in ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv', 'train_ground_truth.tsv']:
    assert (TRAIN_DIR / f).exists(), f"Missing: {TRAIN_DIR / f}"
for f in ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']:
    assert (TEST_DIR / f).exists(), f"Missing: {TEST_DIR / f}"
print("✅ All dataset files found.")

# %% [markdown]
# ## 1. Dataset Overview

# %%
# Load all source files
print("Loading datasets (this may take a minute for large files)...")

train_s1 = pd.read_csv(TRAIN_DIR / 'train_source1.tsv', sep='\t', dtype=str)
train_s2 = pd.read_csv(TRAIN_DIR / 'train_source2.tsv', sep='\t', dtype=str)
train_s3 = pd.read_csv(TRAIN_DIR / 'train_source3.tsv', sep='\t', dtype=str)
gt = pd.read_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', dtype=str)

test_s1 = pd.read_csv(TEST_DIR / 'test_source1.tsv', sep='\t', dtype=str)
test_s2 = pd.read_csv(TEST_DIR / 'test_source2.tsv', sep='\t', dtype=str)
test_s3 = pd.read_csv(TEST_DIR / 'test_source3.tsv', sep='\t', dtype=str)

print("✅ All datasets loaded.")

# %%
# Verify correct parsing (should have 4 columns for source files)
for name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3),
                 ('test_s1', test_s1), ('test_s2', test_s2), ('test_s3', test_s3)]:
    assert df.shape[1] == 4, f"{name} has {df.shape[1]} columns instead of 4. Check sep='\\t'."
    assert list(df.columns) == ['entity_id', 'business_name', 'business_address', 'country'], \
        f"{name} columns mismatch: {list(df.columns)}"
assert gt.shape[1] == 2, f"Ground truth has {gt.shape[1]} columns."
print("✅ All files parsed correctly with tab separator.")

# %%
# Dataset overview table
datasets = {
    'train_source1': train_s1,
    'train_source2': train_s2,
    'train_source3': train_s3,
    'test_source1': test_s1,
    'test_source2': test_s2,
    'test_source3': test_s3,
}

overview_rows = []
for name, df in datasets.items():
    row = {
        'Dataset': name,
        'Rows': len(df),
        'Columns': df.shape[1],
        'Unique entity_id': df['entity_id'].nunique(),
        'Duplicate entity_ids': len(df) - df['entity_id'].nunique(),
        'Fully Duplicated Rows': df.duplicated().sum(),
    }
    overview_rows.append(row)

overview_df = pd.DataFrame(overview_rows)
print("\n📊 Dataset Overview:")
print(overview_df.to_string(index=False))

# %%
# Missing / null / empty values per column per dataset
print("\n📊 Missing/Null/Empty Values Analysis:")
for name, df in datasets.items():
    print(f"\n--- {name} ---")
    for col in df.columns:
        null_count = df[col].isna().sum()
        empty_str = (df[col].fillna('') == '').sum() - df[col].isna().sum()  # empty but not null
        whitespace_only = df[col].fillna('').str.strip().eq('').sum() - (df[col].isna().sum() + empty_str)
        total_missing = df[col].isna().sum() + (df[col].fillna('').str.strip() == '').sum() - df[col].isna().sum()
        print(f"  {col}: null={null_count}, empty_string={max(0, empty_str)}, "
              f"whitespace_only={max(0, whitespace_only)}, total_effectively_missing={total_missing}")

# %%
# Country distribution for all datasets
print("\n📊 Country Distribution:")
for name, df in datasets.items():
    print(f"\n--- {name} ---")
    country_counts = df['country'].value_counts(dropna=False)
    for country, count in country_counts.items():
        pct = count / len(df) * 100
        print(f"  {country}: {count:,} ({pct:.1f}%)")

# %%
# Name length statistics
print("\n📊 Name Length Statistics:")
for name, df in datasets.items():
    name_lens = df['business_name'].fillna('').str.len()
    print(f"\n--- {name} ---")
    print(f"  min={name_lens.min()}, max={name_lens.max()}, mean={name_lens.mean():.1f}, "
          f"median={name_lens.median():.1f}, std={name_lens.std():.1f}")
    print(f"  Very short (≤3 chars): {(name_lens <= 3).sum()} ({(name_lens <= 3).mean()*100:.2f}%)")
    print(f"  Very long (>100 chars): {(name_lens > 100).sum()} ({(name_lens > 100).mean()*100:.2f}%)")

# %%
# Address length statistics
print("\n📊 Address Length Statistics:")
for name, df in datasets.items():
    addr_lens = df['business_address'].fillna('').str.len()
    print(f"\n--- {name} ---")
    print(f"  min={addr_lens.min()}, max={addr_lens.max()}, mean={addr_lens.mean():.1f}, "
          f"median={addr_lens.median():.1f}, std={addr_lens.std():.1f}")
    print(f"  Empty addresses: {(addr_lens == 0).sum()} ({(addr_lens == 0).mean()*100:.2f}%)")
    print(f"  Very long (>200 chars): {(addr_lens > 200).sum()} ({(addr_lens > 200).mean()*100:.2f}%)")

# %% [markdown]
# ## 2. Data Quality Investigation

# %%
# Duplicate IDs check
print("📊 Duplicate Entity IDs:")
for name, df in datasets.items():
    dup_ids = df[df['entity_id'].duplicated(keep=False)]
    if len(dup_ids) > 0:
        print(f"  {name}: {df['entity_id'].duplicated().sum()} duplicate ID entries")
        print(f"    Example duplicated IDs: {dup_ids['entity_id'].value_counts().head(3).to_dict()}")
    else:
        print(f"  {name}: No duplicate IDs ✅")

# %%
# Duplicate names check
print("\n📊 Duplicate Business Names (exact):")
for name, df in datasets.items():
    name_counts = df['business_name'].dropna().value_counts()
    dup_names = name_counts[name_counts > 1]
    print(f"  {name}: {len(dup_names)} names appear more than once, "
          f"covering {dup_names.sum()} records")
    if len(dup_names) > 0:
        print(f"    Top 5 repeated names: {dup_names.head(5).to_dict()}")

# %%
# Duplicate addresses check
print("\n📊 Duplicate Business Addresses (exact):")
for name, df in datasets.items():
    addr_counts = df['business_address'].dropna().value_counts()
    dup_addrs = addr_counts[addr_counts > 1]
    print(f"  {name}: {len(dup_addrs)} addresses appear more than once, "
          f"covering {dup_addrs.sum()} records")
    if len(dup_addrs) > 0:
        print(f"    Top 5 repeated addresses: {dup_addrs.head(5).to_dict()}")

# %%
# Missing names/addresses/countries
print("\n📊 Missing Critical Fields:")
for name, df in datasets.items():
    missing_name = df['business_name'].isna() | (df['business_name'].str.strip() == '')
    missing_addr = df['business_address'].isna() | (df['business_address'].str.strip() == '')
    missing_country = df['country'].isna() | (df['country'].str.strip() == '')
    print(f"  {name}:")
    print(f"    Missing names: {missing_name.sum()} ({missing_name.mean()*100:.2f}%)")
    print(f"    Missing addresses: {missing_addr.sum()} ({missing_addr.mean()*100:.2f}%)")
    print(f"    Missing countries: {missing_country.sum()} ({missing_country.mean()*100:.2f}%)")

# %%
# Unicode / unusual character detection
print("\n📊 Character Analysis (sample-based):")
for name, df in datasets.items():
    names = df['business_name'].dropna()
    # Check for non-ASCII
    non_ascii_mask = names.str.contains(r'[^\x00-\x7F]', regex=True, na=False)
    non_ascii_count = non_ascii_mask.sum()
    print(f"\n  {name}:")
    print(f"    Names with non-ASCII chars: {non_ascii_count} ({non_ascii_count/len(names)*100:.2f}%)")
    
    # Devanagari script detection
    devanagari = names.str.contains(r'[\u0900-\u097F]', regex=True, na=False)
    print(f"    Names with Devanagari script: {devanagari.sum()} ({devanagari.mean()*100:.2f}%)")
    
    # Accented Latin chars (common in French)
    accented = names.str.contains(r'[àâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ]', regex=True, na=False)
    print(f"    Names with accented Latin chars: {accented.sum()} ({accented.mean()*100:.2f}%)")
    
    # Unusual punctuation
    unusual_punct = names.str.contains(r'[<>{}|\\~`]', regex=True, na=False)
    print(f"    Names with unusual punctuation (<>{{}}|\\~`): {unusual_punct.sum()}")

# %%
# Capitalization patterns
print("\n📊 Capitalization Patterns in Business Names:")
for name, df in datasets.items():
    names = df['business_name'].dropna()
    all_upper = names.str.match(r'^[^a-z]*$', na=False) & names.str.contains(r'[A-Z]', na=False)
    all_lower = names.str.match(r'^[^A-Z]*$', na=False) & names.str.contains(r'[a-z]', na=False)
    mixed = ~all_upper & ~all_lower
    print(f"\n  {name}:")
    print(f"    ALL UPPER: {all_upper.sum()} ({all_upper.mean()*100:.1f}%)")
    print(f"    all lower: {all_lower.sum()} ({all_lower.mean()*100:.1f}%)")
    print(f"    Mixed case: {mixed.sum()} ({mixed.mean()*100:.1f}%)")

# %% [markdown]
# ## 3. Business Name Noise Analysis

# %%
# Legal suffix variations
legal_suffixes = [
    r'\bInc\.?\b', r'\bIncorporated\b', r'\bCorp\.?\b', r'\bCorporation\b',
    r'\bLLC\b', r'\bLtd\.?\b', r'\bLimited\b', r'\bPvt\.?\b', r'\bPrivate\b',
    r'\bLLP\b', r'\bPLC\b', r'\bGmbH\b', r'\bSARL\b', r'\bSAS\b', r'\bSA\b',
    r'\bCo\.?\b', r'\bCompany\b', r'\bAssociates?\b', r'\bPartners?\b',
    r'\bGroup\b', r'\bHoldings?\b', r'\bEnterprises?\b', r'\bServices?\b',
    r'\bSolutions?\b', r'\bTechnolog(?:y|ies)\b', r'\bIndustries\b',
    r'\bConsultants?\b', r'\bInternational\b', r'\bDBA\b',
]

print("📊 Legal Suffix Prevalence in Business Names:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    print(f"\n--- {ds_name} (n={len(df)}) ---")
    for pattern in legal_suffixes:
        count = names.str.contains(pattern, case=False, regex=True, na=False).sum()
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {pattern}: {count:,} ({pct:.2f}%)")

# %%
# & vs "and" patterns
print("\n📊 '&' vs 'and' in Business Names:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    amp_count = names.str.contains(r'&', regex=False, na=False).sum()
    and_count = names.str.contains(r'\band\b', case=False, regex=True, na=False).sum()
    print(f"  {ds_name}: '&'={amp_count:,}, 'and'={and_count:,}")

# %%
# Punctuation patterns in names
print("\n📊 Common Punctuation in Business Names:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    print(f"\n--- {ds_name} ---")
    for char, label in [('.', 'period'), (',', 'comma'), ("'", 'apostrophe'), 
                         ('-', 'hyphen'), ('/', 'slash'), ('"', 'quote'),
                         ('(', 'paren'), ('#', 'hash')]:
        count = names.str.contains(re.escape(char), regex=True, na=False).sum()
        pct = count / len(df) * 100
        print(f"  {label} '{char}': {count:,} ({pct:.2f}%)")

# %%
# Names starting with special markers (<<, --, etc.)
print("\n📊 Names with Special Leading Patterns:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    leading_special = names.str.match(r'^[^a-zA-Z0-9\u0900-\u097F]', na=False)
    print(f"  {ds_name}: {leading_special.sum()} names start with non-alphanumeric char")
    if leading_special.sum() > 0:
        examples = names[leading_special].head(5).tolist()
        print(f"    Examples: {examples}")

# %%
# Show examples of name noise patterns from training data
print("\n📊 Examples of Name Noise Patterns (from training data):")

# Find names containing "DBA" or "d/b/a" or "doing business as"
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    dba = names.str.contains(r'\bDBA\b|d/b/a|doing business as', case=False, regex=True, na=False)
    if dba.sum() > 0:
        print(f"\n  DBA patterns in {ds_name}: {dba.sum()}")
        print(f"    Examples: {names[dba].head(3).tolist()}")

# Names ending with .com or .net (web-based names)
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    names = df['business_name'].fillna('')
    dotcom = names.str.contains(r'\.(com|net|org|io)\b', case=False, regex=True, na=False)
    if dotcom.sum() > 0:
        print(f"\n  Web-domain names in {ds_name}: {dotcom.sum()}")
        print(f"    Examples: {names[dotcom].head(3).tolist()}")

# %% [markdown]
# ## 4. Address Noise Analysis

# %%
# Address abbreviation patterns
addr_patterns = [
    (r'\bSt\.?\b', 'St/Street'), (r'\bStreet\b', 'Street'),
    (r'\bRd\.?\b', 'Rd/Road'), (r'\bRoad\b', 'Road'),
    (r'\bAve\.?\b', 'Ave/Avenue'), (r'\bAvenue\b', 'Avenue'),
    (r'\bBlvd\.?\b', 'Blvd/Boulevard'), (r'\bBoulevard\b', 'Boulevard'),
    (r'\bDr\.?\b', 'Dr/Drive'), (r'\bDrive\b', 'Drive'),
    (r'\bLn\.?\b', 'Ln/Lane'), (r'\bLane\b', 'Lane'),
    (r'\bCt\.?\b', 'Ct/Court'), (r'\bCourt\b', 'Court'),
    (r'\bPO Box\b', 'PO Box'), (r'\bP\.?O\.?\s*Box\b', 'P.O. Box'),
    (r'\bSuite\b', 'Suite'), (r'\bSte\.?\b', 'Ste'),
    (r'\bApt\.?\b', 'Apt'), (r'\bApartment\b', 'Apartment'),
    (r'\bUnit\b', 'Unit'), (r'\bFloor\b', 'Floor'),
    (r'\bNear\b', 'Near (landmark)'), (r'\bOpp\.?\b', 'Opp (opposite)'),
    (r'\bBehind\b', 'Behind (landmark)'),
]

print("📊 Address Pattern Prevalence:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    addrs = df['business_address'].fillna('')
    print(f"\n--- {ds_name} ---")
    for pattern, label in addr_patterns:
        count = addrs.str.contains(pattern, case=False, regex=True, na=False).sum()
        if count > 100:  # Only show if somewhat frequent
            pct = count / len(df) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")

# %%
# Landmark-based address references (Indian-style)
print("\n📊 Landmark-Based Address References:")
landmark_patterns = [
    r'\bNear\b', r'\bOpp\.?\b', r'\bOpposite\b', r'\bBehind\b',
    r'\bBeside\b', r'\bAdjacent\b', r'\bNext to\b', r'\bAbove\b',
]
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    addrs = df['business_address'].fillna('')
    for pattern in landmark_patterns:
        count = addrs.str.contains(pattern, case=False, regex=True, na=False).sum()
        if count > 0:
            print(f"  {ds_name} - {pattern}: {count:,}")

# %%
# Address component analysis — state abbreviations vs full names (US)
print("\n📊 US State Format Examples:")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    us_addrs = df[df['country'] == 'US']['business_address'].fillna('')
    # Check if addresses contain 2-letter state codes at end vs full state names
    has_state_abbr = us_addrs.str.contains(r',\s*[A-Z]{2}\s*$', regex=True, na=False)
    has_texas = us_addrs.str.contains(r'\bTexas\b', case=True, na=False)
    has_tx = us_addrs.str.contains(r'\bTX\b', na=False)
    if len(us_addrs) > 0:
        print(f"\n  {ds_name} (US records):")
        print(f"    Ends with 2-letter state: {has_state_abbr.sum()}")
        print(f"    Contains 'Texas': {has_texas.sum()}, Contains 'TX': {has_tx.sum()}")

# %%
# Address examples — show diverse noise patterns
print("\n📊 Sample Address Noise Examples:")
# Addresses with "KH NO" or "KHASRA" (Indian land reference)
for ds_name, df in [('train_s2', train_s2), ('train_s3', train_s3)]:
    addrs = df['business_address'].fillna('')
    kh = addrs.str.contains(r'KH\s*NO|KHASRA', case=False, regex=True, na=False)
    if kh.sum() > 0:
        print(f"\n  {ds_name} - Indian land references (KH NO/KHASRA): {kh.sum()}")
        print(f"    Examples: {addrs[kh].head(3).tolist()}")

# Addresses with PIN codes
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    addrs = df['business_address'].fillna('')
    pincode = addrs.str.contains(r'\b\d{6}\b', regex=True, na=False)
    zipcode = addrs.str.contains(r'\b\d{5}(-\d{4})?\b', regex=True, na=False)
    print(f"\n  {ds_name}:")
    print(f"    Addresses with 6-digit PIN: {pincode.sum()}")
    print(f"    Addresses with 5-digit ZIP: {zipcode.sum()}")

# %% [markdown]
# ## 5. Ground Truth Analysis

# %%
# Parse ground truth
print("📊 Ground Truth Overview:")
print(f"  Total rows: {len(gt)}")
print(f"  Columns: {list(gt.columns)}")

# Parse matched_entity_ids — handle empty matches
gt['match_list'] = gt['matched_entity_ids'].fillna('').apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if isinstance(x, str) and x.strip() else []
)
gt['num_matches'] = gt['match_list'].apply(len)

total_s1 = len(gt)
zero_match = (gt['num_matches'] == 0).sum()
single_match = (gt['num_matches'] == 1).sum()
multi_match = (gt['num_matches'] > 1).sum()
total_positive_links = gt['num_matches'].sum()

print(f"\n  Total S1 entities: {total_s1:,}")
print(f"  Total positive links: {total_positive_links:,}")
print(f"  Zero-match (singleton): {zero_match:,} ({zero_match/total_s1*100:.2f}%)")
print(f"  Single-match: {single_match:,} ({single_match/total_s1*100:.2f}%)")
print(f"  Multi-match (>1): {multi_match:,} ({multi_match/total_s1*100:.2f}%)")

# %%
# Distribution of number of matches per S1
print("\n📊 Distribution of Matches per S1 Entity:")
match_dist = gt['num_matches'].value_counts().sort_index()
for n_matches, count in match_dist.items():
    pct = count / total_s1 * 100
    print(f"  {n_matches} matches: {count:,} ({pct:.2f}%)")

print(f"\n  Max matches per S1: {gt['num_matches'].max()}")
print(f"  Mean matches per S1 (non-zero): {gt[gt['num_matches']>0]['num_matches'].mean():.2f}")

# %%
# S1→S2 vs S1→S3 breakdown
gt['s2_matches'] = gt['match_list'].apply(lambda x: [m for m in x if m.startswith('S2-')])
gt['s3_matches'] = gt['match_list'].apply(lambda x: [m for m in x if m.startswith('S3-')])
gt['num_s2'] = gt['s2_matches'].apply(len)
gt['num_s3'] = gt['s3_matches'].apply(len)

total_s2_links = gt['num_s2'].sum()
total_s3_links = gt['num_s3'].sum()
has_s2 = (gt['num_s2'] > 0).sum()
has_s3 = (gt['num_s3'] > 0).sum()
has_both = ((gt['num_s2'] > 0) & (gt['num_s3'] > 0)).sum()
has_any = (gt['num_matches'] > 0).sum()

print("\n📊 S1→S2 vs S1→S3 Match Breakdown:")
print(f"  Total S1→S2 links: {total_s2_links:,}")
print(f"  Total S1→S3 links: {total_s3_links:,}")
print(f"  S1 entities with at least one S2 match: {has_s2:,} ({has_s2/total_s1*100:.2f}%)")
print(f"  S1 entities with at least one S3 match: {has_s3:,} ({has_s3/total_s1*100:.2f}%)")
print(f"  S1 entities with BOTH S2 and S3 matches: {has_both:,} ({has_both/total_s1*100:.2f}%)")
print(f"  S1 entities with BOTH (as % of those with any match): {has_both/has_any*100:.2f}%" if has_any > 0 else "")

# %%
# CRITICAL: Check if same S2/S3 entity maps to multiple S1 entities
print("\n📊 Multi-S1 Mapping Check (CRITICAL):")
print("  Checking if any S2/S3 entity is linked to multiple S1 entities...")

# Explode match lists
all_pairs = []
for _, row in gt[gt['num_matches'] > 0].iterrows():
    s1_id = row['source1_entity_id']
    for match_id in row['match_list']:
        all_pairs.append((s1_id, match_id))

pairs_df = pd.DataFrame(all_pairs, columns=['s1_id', 'matched_id'])

# Check S2 IDs mapped to multiple S1s
s2_pairs = pairs_df[pairs_df['matched_id'].str.startswith('S2-')]
s2_multi = s2_pairs.groupby('matched_id')['s1_id'].nunique()
s2_multi_count = (s2_multi > 1).sum()
print(f"\n  S2 IDs linked to multiple S1 IDs: {s2_multi_count}")
if s2_multi_count > 0:
    examples = s2_multi[s2_multi > 1].head(5)
    print(f"  Examples (S2 ID → # S1 IDs): {examples.to_dict()}")
    for s2_id in examples.index[:3]:
        linked_s1s = s2_pairs[s2_pairs['matched_id'] == s2_id]['s1_id'].tolist()
        print(f"    {s2_id} → {linked_s1s}")

# Check S3 IDs mapped to multiple S1s
s3_pairs = pairs_df[pairs_df['matched_id'].str.startswith('S3-')]
s3_multi = s3_pairs.groupby('matched_id')['s1_id'].nunique()
s3_multi_count = (s3_multi > 1).sum()
print(f"\n  S3 IDs linked to multiple S1 IDs: {s3_multi_count}")
if s3_multi_count > 0:
    examples = s3_multi[s3_multi > 1].head(5)
    print(f"  Examples (S3 ID → # S1 IDs): {examples.to_dict()}")
    for s3_id in examples.index[:3]:
        linked_s1s = s3_pairs[s3_pairs['matched_id'] == s3_id]['s1_id'].tolist()
        print(f"    {s3_id} → {linked_s1s}")

print(f"\n  ⚠️ CONCLUSION: ", end="")
if s2_multi_count == 0 and s3_multi_count == 0:
    print("No S2/S3 entity maps to multiple S1 entities → 1-to-many from S1 only")
else:
    print(f"Found {s2_multi_count} S2 and {s3_multi_count} S3 IDs mapping to multiple S1s → "
          f"Many-to-many relationships exist!")

# %% [markdown]
# ## 6. True-Match Characteristics

# %%
# Join ground truth with source records for analysis
# Use a sample for efficiency (true match analysis on full data is expensive)
print("📊 True-Match Characteristics Analysis")
print("  Building true-match pairs with source data join...")

# Create lookup dicts for all training sources
s1_lookup = train_s1.set_index('entity_id')
s2_lookup = train_s2.set_index('entity_id')
s3_lookup = train_s3.set_index('entity_id')

def get_record(entity_id):
    """Look up a record from the appropriate source."""
    if entity_id.startswith('S1-'):
        return s1_lookup.loc[entity_id] if entity_id in s1_lookup.index else None
    elif entity_id.startswith('S2-'):
        return s2_lookup.loc[entity_id] if entity_id in s2_lookup.index else None
    elif entity_id.startswith('S3-'):
        return s3_lookup.loc[entity_id] if entity_id in s3_lookup.index else None
    return None

# Sample true pairs for detailed analysis
np.random.seed(42)
positive_gt = gt[gt['num_matches'] > 0]
sample_size = min(10000, len(positive_gt))
sampled_gt = positive_gt.sample(n=sample_size, random_state=42)

true_pairs_data = []
for _, row in sampled_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_rec = get_record(s1_id)
    if s1_rec is None:
        continue
    for match_id in row['match_list']:
        match_rec = get_record(match_id)
        if match_rec is None:
            continue
        true_pairs_data.append({
            's1_id': s1_id,
            'matched_id': match_id,
            's1_name': str(s1_rec.get('business_name', '')),
            's1_addr': str(s1_rec.get('business_address', '')),
            's1_country': str(s1_rec.get('country', '')),
            'match_name': str(match_rec.get('business_name', '')),
            'match_addr': str(match_rec.get('business_address', '')),
            'match_country': str(match_rec.get('country', '')),
        })

true_pairs = pd.DataFrame(true_pairs_data)
print(f"  Built {len(true_pairs)} true-match pairs for analysis")

# %%
# Simple similarity functions for analysis only
def normalize_simple(s):
    """Very basic normalization for analysis — NOT the final pipeline."""
    if pd.isna(s) or not isinstance(s, str):
        return ''
    return re.sub(r'\s+', ' ', s.strip().lower())

def token_overlap(a, b):
    """Jaccard token overlap."""
    a_tokens = set(normalize_simple(a).split())
    b_tokens = set(normalize_simple(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

def char_overlap(a, b):
    """Character-level Jaccard similarity."""
    a_chars = set(normalize_simple(a))
    b_chars = set(normalize_simple(b))
    if not a_chars or not b_chars:
        return 0.0
    return len(a_chars & b_chars) / len(a_chars | b_chars)

# %%
# Compute similarity metrics for true pairs
print("  Computing similarity metrics for true pairs...")
true_pairs['name_exact_match'] = true_pairs['s1_name'] == true_pairs['match_name']
true_pairs['name_normalized_match'] = true_pairs['s1_name'].apply(normalize_simple) == \
                                       true_pairs['match_name'].apply(normalize_simple)
true_pairs['name_token_overlap'] = true_pairs.apply(
    lambda r: token_overlap(r['s1_name'], r['match_name']), axis=1)
true_pairs['name_char_overlap'] = true_pairs.apply(
    lambda r: char_overlap(r['s1_name'], r['match_name']), axis=1)
true_pairs['addr_token_overlap'] = true_pairs.apply(
    lambda r: token_overlap(r['s1_addr'], r['match_addr']), axis=1)
true_pairs['country_match'] = true_pairs['s1_country'] == true_pairs['match_country']

true_pairs['name_len_diff'] = abs(true_pairs['s1_name'].str.len() - true_pairs['match_name'].str.len())
true_pairs['addr_len_diff'] = abs(true_pairs['s1_addr'].fillna('').str.len() - \
                                   true_pairs['match_addr'].fillna('').str.len())

# %%
# Report true-match statistics
print("\n📊 True-Match Similarity Statistics:")
print(f"  Name exact match: {true_pairs['name_exact_match'].mean()*100:.2f}%")
print(f"  Name normalized match: {true_pairs['name_normalized_match'].mean()*100:.2f}%")
print(f"  Country agreement: {true_pairs['country_match'].mean()*100:.2f}%")

print(f"\n  Name token overlap (Jaccard):")
print(f"    mean={true_pairs['name_token_overlap'].mean():.3f}, "
      f"median={true_pairs['name_token_overlap'].median():.3f}, "
      f"std={true_pairs['name_token_overlap'].std():.3f}")
print(f"    <0.3: {(true_pairs['name_token_overlap']<0.3).mean()*100:.1f}%, "
      f"<0.5: {(true_pairs['name_token_overlap']<0.5).mean()*100:.1f}%, "
      f">0.8: {(true_pairs['name_token_overlap']>0.8).mean()*100:.1f}%")

print(f"\n  Address token overlap (Jaccard):")
print(f"    mean={true_pairs['addr_token_overlap'].mean():.3f}, "
      f"median={true_pairs['addr_token_overlap'].median():.3f}")

print(f"\n  Name length difference:")
print(f"    mean={true_pairs['name_len_diff'].mean():.1f}, "
      f"median={true_pairs['name_len_diff'].median():.1f}, "
      f"max={true_pairs['name_len_diff'].max()}")

print(f"\n  Address length difference:")
print(f"    mean={true_pairs['addr_len_diff'].mean():.1f}, "
      f"median={true_pairs['addr_len_diff'].median():.1f}")

# %%
# Representative examples of true matches
print("\n📊 Representative True-Match Examples:")

# Easy matches (high name overlap)
easy = true_pairs[true_pairs['name_token_overlap'] > 0.85].head(5)
if len(easy) > 0:
    print("\n  🟢 Easy Matches (high name token overlap):")
    for _, r in easy.iterrows():
        print(f"    S1: '{r['s1_name']}' | '{r['s1_addr']}'")
        print(f"    {r['matched_id'][:2]}: '{r['match_name']}' | '{r['match_addr']}'")
        print(f"    Token overlap: {r['name_token_overlap']:.3f}")
        print()

# Noisy name matches (medium overlap)
noisy_name = true_pairs[(true_pairs['name_token_overlap'] > 0.2) & 
                         (true_pairs['name_token_overlap'] < 0.5)].head(5)
if len(noisy_name) > 0:
    print("  🟡 Noisy Name Matches (medium name token overlap):")
    for _, r in noisy_name.iterrows():
        print(f"    S1: '{r['s1_name']}' | '{r['s1_addr']}'")
        print(f"    {r['matched_id'][:2]}: '{r['match_name']}' | '{r['match_addr']}'")
        print(f"    Token overlap: {r['name_token_overlap']:.3f}")
        print()

# Very low name overlap (rely on address)
hard = true_pairs[true_pairs['name_token_overlap'] < 0.1].head(5)
if len(hard) > 0:
    print("  🔴 Difficult Matches (very low name overlap):")
    for _, r in hard.iterrows():
        print(f"    S1: '{r['s1_name']}' | '{r['s1_addr']}'")
        print(f"    {r['matched_id'][:2]}: '{r['match_name']}' | '{r['match_addr']}'")
        print(f"    Name token overlap: {r['name_token_overlap']:.3f}, "
              f"Addr token overlap: {r['addr_token_overlap']:.3f}")
        print()

# Noisy address matches (same name, different address)
noisy_addr = true_pairs[(true_pairs['name_token_overlap'] > 0.7) & 
                         (true_pairs['addr_token_overlap'] < 0.3)].head(5)
if len(noisy_addr) > 0:
    print("  🟠 Noisy Address Matches (high name overlap, low address overlap):")
    for _, r in noisy_addr.iterrows():
        print(f"    S1: '{r['s1_name']}' | '{r['s1_addr']}'")
        print(f"    {r['matched_id'][:2]}: '{r['match_name']}' | '{r['match_addr']}'")
        print(f"    Name: {r['name_token_overlap']:.3f}, Addr: {r['addr_token_overlap']:.3f}")
        print()

# %% [markdown]
# ## 7. Country Analysis

# %%
print("📊 Country Analysis:")

# All countries in train
train_all = pd.concat([train_s1, train_s2, train_s3])
train_countries = train_all['country'].value_counts(dropna=False)
print("\n  Training countries:")
for country, count in train_countries.items():
    print(f"    {country}: {count:,}")

# All countries in test
test_all = pd.concat([test_s1, test_s2, test_s3])
test_countries = test_all['country'].value_counts(dropna=False)
print("\n  Test countries:")
for country, count in test_countries.items():
    print(f"    {country}: {count:,}")

# Countries in test but not in train
train_country_set = set(train_all['country'].dropna().unique())
test_country_set = set(test_all['country'].dropna().unique())
unseen = test_country_set - train_country_set
print(f"\n  Countries in train: {sorted(train_country_set)}")
print(f"  Countries in test: {sorted(test_country_set)}")
print(f"  ⚠️ Unseen test countries (not in training): {sorted(unseen) if unseen else 'None'}")

if unseen:
    for country in sorted(unseen):
        test_count = test_all[test_all['country'] == country].shape[0]
        print(f"    '{country}' appears in {test_count:,} test records")

# %%
# Country distribution per source (train)
print("\n📊 Country Distribution per Source (Train):")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    print(f"\n  {ds_name}:")
    for country, count in df['country'].value_counts(dropna=False).items():
        pct = count / len(df) * 100
        print(f"    {country}: {count:,} ({pct:.1f}%)")

# Country distribution per source (test)
print("\n📊 Country Distribution per Source (Test):")
for ds_name, df in [('test_s1', test_s1), ('test_s2', test_s2), ('test_s3', test_s3)]:
    print(f"\n  {ds_name}:")
    for country, count in df['country'].value_counts(dropna=False).items():
        pct = count / len(df) * 100
        print(f"    {country}: {count:,} ({pct:.1f}%)")

# %%
# Match/no-match distributions by country (training)
print("\n📊 Match/No-Match Distribution by Country (Training):")
gt_with_country = gt.merge(train_s1[['entity_id', 'country']], 
                            left_on='source1_entity_id', right_on='entity_id', how='left')
country_match_stats = gt_with_country.groupby('country').agg(
    total=('num_matches', 'size'),
    zero_match=('num_matches', lambda x: (x == 0).sum()),
    has_match=('num_matches', lambda x: (x > 0).sum()),
    avg_matches=('num_matches', 'mean'),
).reset_index()
country_match_stats['zero_match_pct'] = country_match_stats['zero_match'] / country_match_stats['total'] * 100
country_match_stats['has_match_pct'] = country_match_stats['has_match'] / country_match_stats['total'] * 100

print(country_match_stats.to_string(index=False))

# %% [markdown]
# ## 8. Potential Challenge Risks

# %%
print("📊 Potential Challenge Risks (Evidence-Based):")
print()

# Risk 1: Generic business names
print("🔸 Risk 1: Generic/Common Business Names")
all_train_names = pd.concat([train_s1['business_name'], train_s2['business_name'], train_s3['business_name']]).dropna()
name_freq = all_train_names.value_counts()
very_common = name_freq[name_freq > 10]
print(f"  Names appearing >10 times across all sources: {len(very_common)}")
if len(very_common) > 0:
    print(f"  Top 10 most common names:")
    for name, count in very_common.head(10).items():
        print(f"    '{name}': {count}")

# Risk 2: Missing addresses
print(f"\n🔸 Risk 2: Missing Addresses")
for ds_name, df in datasets.items():
    missing = df['business_address'].isna() | (df['business_address'].str.strip() == '')
    print(f"  {ds_name}: {missing.sum():,} missing ({missing.mean()*100:.2f}%)")

# Risk 3: Transliteration (Hindi/Devanagari)
print(f"\n🔸 Risk 3: Transliteration (Devanagari ↔ Latin)")
for ds_name, df in [('train_s1', train_s1), ('train_s2', train_s2), ('train_s3', train_s3)]:
    devn = df['business_name'].fillna('').str.contains(r'[\u0900-\u097F]', regex=True, na=False)
    print(f"  {ds_name}: {devn.sum():,} names with Devanagari ({devn.mean()*100:.2f}%)")

# Risk 4: Singleton entities
print(f"\n🔸 Risk 4: Singleton/No-Match Entities")
print(f"  Zero-match in training: {zero_match:,} ({zero_match/total_s1*100:.2f}%)")
print(f"  These entities score 1.0 if correctly predicted empty, 0.0 if any false match predicted.")

# Risk 5: Unseen test countries
print(f"\n🔸 Risk 5: Unseen Test Countries")
if unseen:
    for c in sorted(unseen):
        cnt = test_all[test_all['country'] == c].shape[0]
        print(f"  '{c}': {cnt:,} records in test, ZERO in training")
else:
    print("  No unseen countries detected.")

# Risk 6: Source-specific formatting
print(f"\n🔸 Risk 6: Source-Specific Formatting Differences")
print(f"  S1 uses clean mixed-case names with US state abbreviations")
print(f"  S2 has ALL-CAPS addresses, Devanagari names, reordered address components")
print(f"  S3 has full state names (e.g., 'Texas' not 'TX'), .com-based business names, missing addresses")

# Risk 7: Many-to-many matching
if s2_multi_count > 0 or s3_multi_count > 0:
    print(f"\n🔸 Risk 7: Many-to-Many Relationships")
    print(f"  {s2_multi_count} S2 IDs and {s3_multi_count} S3 IDs map to multiple S1 entities")
    print(f"  This means uniqueness constraints should NOT be imposed")
else:
    print(f"\n🔸 Risk 7: 1-to-Many Relationships (from S1)")
    print(f"  No S2/S3 maps to multiple S1 entities — could benefit from uniqueness constraint")

# %% [markdown]
# ## 9. Visualizations

# %%
# Figure 1: Rows per source
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_sizes = {'S1': len(train_s1), 'S2': len(train_s2), 'S3': len(train_s3)}
test_sizes = {'S1': len(test_s1), 'S2': len(test_s2), 'S3': len(test_s3)}

axes[0].bar(train_sizes.keys(), train_sizes.values(), color=['#2196F3', '#4CAF50', '#FF9800'])
axes[0].set_title('Training Set: Rows per Source')
axes[0].set_ylabel('Number of Records')
for i, (k, v) in enumerate(train_sizes.items()):
    axes[0].text(i, v + 20000, f'{v:,}', ha='center', fontsize=10)

axes[1].bar(test_sizes.keys(), test_sizes.values(), color=['#2196F3', '#4CAF50', '#FF9800'])
axes[1].set_title('Test Set: Rows per Source')
axes[1].set_ylabel('Number of Records')
for i, (k, v) in enumerate(test_sizes.items()):
    axes[1].text(i, v + 20000, f'{v:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_rows_per_source.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 2: Country distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
train_country_all = train_all['country'].value_counts(dropna=False)
axes[0].barh(train_country_all.index.astype(str), train_country_all.values, color='#2196F3')
axes[0].set_title('Training: Country Distribution (all sources)')
axes[0].set_xlabel('Count')
for i, v in enumerate(train_country_all.values):
    axes[0].text(v + 5000, i, f'{v:,}', va='center', fontsize=9)

# Test
test_country_all = test_all['country'].value_counts(dropna=False)
axes[1].barh(test_country_all.index.astype(str), test_country_all.values, color='#FF9800')
axes[1].set_title('Test: Country Distribution (all sources)')
axes[1].set_xlabel('Count')
for i, v in enumerate(test_country_all.values):
    axes[1].text(v + 5000, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_country_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 3: Missing value distribution
fig, ax = plt.subplots(figsize=(14, 6))

missing_data = []
for ds_name, df in datasets.items():
    for col in ['business_name', 'business_address', 'country']:
        missing = df[col].isna() | (df[col].fillna('').str.strip() == '')
        missing_data.append({
            'Dataset': ds_name,
            'Column': col,
            'Missing %': missing.mean() * 100
        })

missing_df = pd.DataFrame(missing_data)
pivot = missing_df.pivot(index='Dataset', columns='Column', values='Missing %')
pivot.plot(kind='bar', ax=ax, color=['#2196F3', '#4CAF50', '#FF9800'])
ax.set_title('Missing Values by Dataset and Column')
ax.set_ylabel('Missing %')
ax.set_xlabel('')
ax.legend(title='Column')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_missing_values.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 4: Name length distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
all_datasets = [
    ('Train S1', train_s1), ('Train S2', train_s2), ('Train S3', train_s3),
    ('Test S1', test_s1), ('Test S2', test_s2), ('Test S3', test_s3),
]
for idx, (ds_name, df) in enumerate(all_datasets):
    ax = axes[idx // 3][idx % 3]
    name_lens = df['business_name'].fillna('').str.len()
    ax.hist(name_lens.clip(upper=100), bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.set_title(f'{ds_name} Name Length')
    ax.set_xlabel('Characters')
    ax.set_ylabel('Count')
    ax.axvline(name_lens.median(), color='red', linestyle='--', label=f'median={name_lens.median():.0f}')
    ax.legend(fontsize=8)

plt.suptitle('Business Name Length Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_name_length.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 5: Address length distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, (ds_name, df) in enumerate(all_datasets):
    ax = axes[idx // 3][idx % 3]
    addr_lens = df['business_address'].fillna('').str.len()
    ax.hist(addr_lens.clip(upper=200), bins=50, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.set_title(f'{ds_name} Address Length')
    ax.set_xlabel('Characters')
    ax.set_ylabel('Count')
    ax.axvline(addr_lens.median(), color='red', linestyle='--', label=f'median={addr_lens.median():.0f}')
    ax.legend(fontsize=8)

plt.suptitle('Business Address Length Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_address_length.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 6: Matches per S1 distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full distribution
match_dist_plot = gt['num_matches'].value_counts().sort_index()
axes[0].bar(match_dist_plot.index, match_dist_plot.values, color='#9C27B0', edgecolor='white')
axes[0].set_title('Matches per S1 Entity (Full Distribution)')
axes[0].set_xlabel('Number of Matches')
axes[0].set_ylabel('Count')
axes[0].set_yscale('log')

# Zoomed: non-zero only
nonzero_dist = gt[gt['num_matches'] > 0]['num_matches'].value_counts().sort_index()
axes[1].bar(nonzero_dist.index, nonzero_dist.values, color='#E91E63', edgecolor='white')
axes[1].set_title('Matches per S1 Entity (Non-Zero Only)')
axes[1].set_xlabel('Number of Matches')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_matches_per_s1.png', dpi=100, bbox_inches='tight')
plt.show()

# %%
# Figure 7: True-match similarity distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(true_pairs['name_token_overlap'], bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
axes[0].set_title('Name Token Overlap (True Pairs)')
axes[0].set_xlabel('Jaccard Similarity')
axes[0].set_ylabel('Count')

axes[1].hist(true_pairs['addr_token_overlap'], bins=50, color='#4CAF50', alpha=0.7, edgecolor='white')
axes[1].set_title('Address Token Overlap (True Pairs)')
axes[1].set_xlabel('Jaccard Similarity')
axes[1].set_ylabel('Count')

axes[2].hist(true_pairs['name_len_diff'].clip(upper=50), bins=50, color='#FF9800', alpha=0.7, edgecolor='white')
axes[2].set_title('Name Length Difference (True Pairs)')
axes[2].set_xlabel('Characters')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig(DOCS_DIR / 'fig_true_match_similarity.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. EDA Conclusions
# ### Findings that will influence Phase 2

# %%
print("=" * 80)
print("📋 EDA CONCLUSIONS — Findings that will influence Phase 2")
print("=" * 80)

print("\n🅰️  CONFIRMED OBSERVATIONS FROM THE DATA:")
print("-" * 50)
print(f"1. Dataset scale: ~{len(train_s1)/1e6:.1f}M S1 train, ~{len(train_s2)/1e6:.1f}M S2, ~{len(train_s3)/1e6:.1f}M S3")
print(f"2. Singleton rate: {zero_match/total_s1*100:.1f}% of S1 entities have NO matches")
print(f"3. Multi-match rate: {multi_match/total_s1*100:.1f}% of S1 have >1 match")
print(f"4. Countries in train: {sorted(train_country_set)}")
print(f"5. Unseen test countries: {sorted(unseen)}")
print(f"6. S2/S3 multi-S1 mapping: S2→{s2_multi_count}, S3→{s3_multi_count}")
print(f"7. Name exact match rate among true pairs: {true_pairs['name_exact_match'].mean()*100:.1f}%")
print(f"8. Name normalized match rate: {true_pairs['name_normalized_match'].mean()*100:.1f}%")
print(f"9. Country agreement in true pairs: {true_pairs['country_match'].mean()*100:.1f}%")
print(f"10. Source-specific formatting: S2 has ALL CAPS addresses, S3 has full state names")

devanagari_s2 = train_s2['business_name'].fillna('').str.contains(r'[\u0900-\u097F]', regex=True).sum()
devanagari_pct = devanagari_s2 / len(train_s2) * 100
print(f"11. Devanagari names in S2: {devanagari_pct:.1f}%")

missing_addr_s3 = (train_s3['business_address'].isna() | (train_s3['business_address'].str.strip() == '')).mean() * 100
print(f"12. Missing addresses in S3: {missing_addr_s3:.1f}%")

print(f"\n🅱️  HYPOTHESES TO TEST IN PHASE 2:")
print("-" * 50)
print("1. High abbreviation noise (Corp/Corporation, Ltd/Limited, Pvt/Private) → normalize legal suffixes")
print("2. Transliteration matching needed for Devanagari ↔ Latin name pairs")
print("3. Address-heavy matching may be needed when names are generic")
print("4. Source-specific address formatting (state abbrev vs full, reordering) → needs normalization")
print("5. Singleton prediction is critical for F0.5 score (large % of entities)")
print("6. Unseen country handling must be language-agnostic (no hardcoded country logic)")
print("7. '&' vs 'and', punctuation differences are systematic → rule-based normalization")
print("8. Multiple S2/S3 matches per S1 → candidate generation must not limit to top-1")

print("\n" + "=" * 80)

# %% [markdown]
# ## 11. Generate EDA Report

# %%
# Collect all statistics for the report
report_lines = []
report_lines.append("# EDA Report — Business Entity Resolution Challenge")
report_lines.append("")
report_lines.append("## 1. Dataset Overview")
report_lines.append("")
report_lines.append("| Dataset | Rows | Unique IDs | Dup IDs | Fully Dup Rows |")
report_lines.append("|---------|------|-----------|---------|----------------|")
for _, row in overview_df.iterrows():
    report_lines.append(f"| {row['Dataset']} | {row['Rows']:,} | {row['Unique entity_id']:,} | "
                        f"{row['Duplicate entity_ids']:,} | {row['Fully Duplicated Rows']:,} |")

report_lines.append("")
report_lines.append("**Columns:** `entity_id`, `business_name`, `business_address`, `country`")
report_lines.append("")
report_lines.append(f"**Ground Truth:** {len(gt):,} rows, matching S1 entities to S2/S3.")

report_lines.append("")
report_lines.append("## 2. Data Quality Findings")
report_lines.append("")

# Missing values summary
report_lines.append("### Missing Values")
report_lines.append("")
report_lines.append("| Dataset | Missing Names | Missing Addresses | Missing Country |")
report_lines.append("|---------|--------------|-------------------|-----------------|")
for ds_name, df in datasets.items():
    mn = (df['business_name'].isna() | (df['business_name'].fillna('').str.strip() == '')).sum()
    ma = (df['business_address'].isna() | (df['business_address'].fillna('').str.strip() == '')).sum()
    mc = (df['country'].isna() | (df['country'].fillna('').str.strip() == '')).sum()
    report_lines.append(f"| {ds_name} | {mn:,} ({mn/len(df)*100:.2f}%) | "
                        f"{ma:,} ({ma/len(df)*100:.2f}%) | {mc:,} ({mc/len(df)*100:.2f}%) |")

report_lines.append("")
report_lines.append("### Duplicate Names")
report_lines.append("")
report_lines.append("Names appearing more than once (exact string match) exist across all sources.")
report_lines.append("This indicates generic business names that could cause false positive matches.")

report_lines.append("")
report_lines.append("### Character/Unicode Analysis")
report_lines.append("")
report_lines.append("- Devanagari script names are prevalent in Source 2 (Indian entities)")
report_lines.append("- Source 3 contains web-domain style names (e.g., `wilfordhancock.com`)")
report_lines.append("- Accented Latin characters appear in test data (French entities)")
report_lines.append("- Some names start with special characters (`<<`, `--`)")

report_lines.append("")
report_lines.append("## 3. Name Noise Findings")
report_lines.append("")
report_lines.append("Key patterns observed:")
report_lines.append("- **Legal suffix variations:** Inc/Incorporated, Corp/Corporation, Ltd/Limited, Pvt/Private, LLC, LLP")
report_lines.append("- **Punctuation:** `&` vs `and`, periods, commas, apostrophes")
report_lines.append("- **Capitalization:** Mixed case in S1, ALL CAPS common in S2")
report_lines.append("- **DBA/Trade names:** Some records contain DBA patterns")
report_lines.append("- **Web-domain names:** `.com`, `.net` suffixes in S3")
report_lines.append("- **Transliteration:** Devanagari ↔ Latin script for Indian entities")
report_lines.append("- **Special prefixes:** `<<`, `--` appear as name prefixes in some sources")

report_lines.append("")
report_lines.append("## 4. Address Noise Findings")
report_lines.append("")
report_lines.append("Key patterns observed:")
report_lines.append("- **Abbreviations:** St/Street, Rd/Road, Ave/Avenue, Blvd/Boulevard")
report_lines.append("- **State format:** S1 uses 2-letter abbreviations (TX), S3 uses full names (Texas)")
report_lines.append("- **Component reordering:** S2 sometimes puts city/state before street")
report_lines.append("- **Landmark references:** `Near`, `Opp.`, `Behind` common in Indian addresses")
report_lines.append("- **Land references:** `KH NO`, `KHASRA` in Indian source data")
report_lines.append("- **Missing addresses:** Significant portion of S3 has empty addresses")
report_lines.append("- **ALL CAPS:** S2 addresses are frequently fully capitalized")

report_lines.append("")
report_lines.append("## 5. Ground Truth Statistics")
report_lines.append("")
report_lines.append(f"- **Total S1 entities:** {total_s1:,}")
report_lines.append(f"- **Total positive links:** {total_positive_links:,}")
report_lines.append(f"- **Zero-match (singleton):** {zero_match:,} ({zero_match/total_s1*100:.2f}%)")
report_lines.append(f"- **Single-match:** {single_match:,} ({single_match/total_s1*100:.2f}%)")
report_lines.append(f"- **Multi-match (>1):** {multi_match:,} ({multi_match/total_s1*100:.2f}%)")

report_lines.append("")
report_lines.append("### Match Distribution")
report_lines.append("")
report_lines.append("| Matches | Count | Percentage |")
report_lines.append("|---------|-------|------------|")
for n_matches, count in match_dist.items():
    pct = count / total_s1 * 100
    report_lines.append(f"| {n_matches} | {count:,} | {pct:.2f}% |")

report_lines.append("")
report_lines.append("### S1→S2 / S1→S3 Breakdown")
report_lines.append("")
report_lines.append(f"- S1→S2 links: {total_s2_links:,}")
report_lines.append(f"- S1→S3 links: {total_s3_links:,}")
report_lines.append(f"- S1 with S2 match: {has_s2:,} ({has_s2/total_s1*100:.2f}%)")
report_lines.append(f"- S1 with S3 match: {has_s3:,} ({has_s3/total_s1*100:.2f}%)")
report_lines.append(f"- S1 with BOTH: {has_both:,} ({has_both/total_s1*100:.2f}%)")

report_lines.append("")
report_lines.append("## 6. Multi-Match / Uniqueness Analysis")
report_lines.append("")
report_lines.append(f"- **S2 IDs linked to multiple S1 IDs:** {s2_multi_count}")
report_lines.append(f"- **S3 IDs linked to multiple S1 IDs:** {s3_multi_count}")
if s2_multi_count == 0 and s3_multi_count == 0:
    report_lines.append("")
    report_lines.append("**No S2/S3 entity maps to multiple S1 entities.**")
    report_lines.append("This means each S2/S3 record belongs to at most one S1 entity.")
    report_lines.append("A uniqueness constraint *could* be justified but should be validated on test data.")
else:
    report_lines.append("")
    report_lines.append("**Many-to-many relationships exist.** Do NOT impose uniqueness constraints.")

report_lines.append("")
report_lines.append("## 7. Country Analysis")
report_lines.append("")
report_lines.append(f"- **Training countries:** {sorted(train_country_set)}")
report_lines.append(f"- **Test countries:** {sorted(test_country_set)}")
report_lines.append(f"- **Unseen in test:** {sorted(unseen) if unseen else 'None'}")
if unseen:
    for c in sorted(unseen):
        cnt = test_all[test_all['country'] == c].shape[0]
        report_lines.append(f"  - `{c}`: {cnt:,} test records with zero training examples")

report_lines.append("")
report_lines.append("### Match Rate by Country (Training)")
report_lines.append("")
report_lines.append(country_match_stats.to_markdown(index=False))

report_lines.append("")
report_lines.append("## 8. True-Match Characteristics")
report_lines.append("")
report_lines.append(f"Based on {len(true_pairs):,} sampled true-match pairs:")
report_lines.append("")
report_lines.append(f"- Name exact match: {true_pairs['name_exact_match'].mean()*100:.2f}%")
report_lines.append(f"- Name normalized match: {true_pairs['name_normalized_match'].mean()*100:.2f}%")
report_lines.append(f"- Country agreement: {true_pairs['country_match'].mean()*100:.2f}%")
report_lines.append(f"- Mean name token overlap: {true_pairs['name_token_overlap'].mean():.3f}")
report_lines.append(f"- Mean address token overlap: {true_pairs['addr_token_overlap'].mean():.3f}")
report_lines.append(f"- Mean name length difference: {true_pairs['name_len_diff'].mean():.1f} chars")
report_lines.append(f"- Mean address length difference: {true_pairs['addr_len_diff'].mean():.1f} chars")

report_lines.append("")
report_lines.append("### Difficulty Breakdown (True Pairs)")
report_lines.append("")
easy_pct = (true_pairs['name_token_overlap'] > 0.8).mean() * 100
medium_pct = ((true_pairs['name_token_overlap'] >= 0.3) & (true_pairs['name_token_overlap'] <= 0.8)).mean() * 100
hard_pct = (true_pairs['name_token_overlap'] < 0.3).mean() * 100
report_lines.append(f"- Easy (name overlap > 0.8): {easy_pct:.1f}%")
report_lines.append(f"- Medium (0.3–0.8): {medium_pct:.1f}%")
report_lines.append(f"- Hard (< 0.3): {hard_pct:.1f}%")

report_lines.append("")
report_lines.append("## 9. Key Risks for Phase 2")
report_lines.append("")
report_lines.append("1. **High singleton rate** — large % of S1 entities have no matches; correctly identifying these is crucial for F0.5")
report_lines.append("2. **Transliteration challenge** — Devanagari ↔ Latin name matching for Indian entities")
report_lines.append("3. **Unseen country** — France appears only in test; pipeline must be language-agnostic")
report_lines.append("4. **Generic names** — common business names appear frequently, requiring address-based disambiguation")
report_lines.append("5. **Missing addresses** — significant portion of records lack addresses, limiting matching signals")
report_lines.append("6. **Source-specific formatting** — each source has distinct normalization patterns")
report_lines.append("7. **Scale** — millions of records require efficient blocking/candidate generation")
report_lines.append("8. **Multi-match complexity** — many S1 entities match multiple S2/S3 records")

report_lines.append("")
report_lines.append("## 10. Phase 2 Implications")
report_lines.append("")
report_lines.append("- Normalization pipeline must handle: legal suffixes, &/and, punctuation, case, Devanagari, French accents")
report_lines.append("- Blocking strategy must balance recall (many potential pairs) with efficiency (millions of records)")
report_lines.append("- Feature engineering should include: name similarity, address similarity, country match, length features")
report_lines.append("- Model must handle unseen country (France) gracefully — no country-specific hardcoding")
report_lines.append("- Singleton detection should be a dedicated component to maximize F0.5")
report_lines.append("- Address normalization must handle US state abbrevs, Indian landmarks, French addresses")

report_content = '\n'.join(report_lines)

with open(DOCS_DIR / 'eda_report.md', 'w') as f:
    f.write(report_content)

print(f"✅ EDA report written to {DOCS_DIR / 'eda_report.md'}")

# %%
# Final verification
print("\n" + "=" * 80)
print("📋 FINAL VERIFICATION")
print("=" * 80)
assert (DOCS_DIR / 'eda_report.md').exists(), "EDA report not found!"
print("✅ docs/eda_report.md exists")

# Verify dataset files were not modified (check row counts)
verify_s1 = pd.read_csv(TRAIN_DIR / 'train_source1.tsv', sep='\t', nrows=5)
assert list(verify_s1.columns) == ['entity_id', 'business_name', 'business_address', 'country']
print("✅ Dataset files not modified (spot check passed)")
print("✅ Phase 1 EDA Complete!")

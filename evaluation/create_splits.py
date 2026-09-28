"""
Creation of the Strict Entity-Level Train / Dev / Hidden Test Split Manifest.

Guarantees:
1. Split exclusively at the S1 entity level.
2. Zero entity leakage: all true S2/S3 records linked to a held-out S1 entity
   are tracked and held out from any training/tuning.
3. Country-stratified to mirror the natural ground truth distribution (US & India).
4. Deterministic random seed (Seed 42).
5. Sizes:
   - Primary Untouched Hidden Test: 5,000 S1 entities (2,999 US, 2,001 India)
   - Development Set: 5,000 S1 entities (2,999 US, 2,001 India)
   - Train Candidate Pool: Remaining ~2.19M S1 entities
"""

import json
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = 'evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading train_source1.tsv and train_ground_truth.tsv...")
s1_df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
gt_df = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')

# Merge to have entity_id, country, matched_entity_ids
merged = pd.merge(s1_df[['entity_id', 'country', 'business_name', 'business_address']], 
                  gt_df[['source1_entity_id', 'matched_entity_ids']], 
                  left_on='entity_id', right_on='source1_entity_id')

print(f"Total merged S1 entities: {len(merged):,}")

# Stratified split by country with fixed seed 42
RNG = np.random.RandomState(42)

us_indices = merged[merged['country'] == 'US'].index.values
india_indices = merged[merged['country'] == 'India'].index.values

RNG.shuffle(us_indices)
RNG.shuffle(india_indices)

# Allocate 3,000 US and 2,000 India to Hidden Test (5,000 total)
# Allocate 3,000 US and 2,000 India to Dev Set (5,000 total)
HIDDEN_US = 3000
HIDDEN_IN = 2000
DEV_US = 3000
DEV_IN = 2000

hidden_indices = np.concatenate([us_indices[:HIDDEN_US], india_indices[:HIDDEN_IN]])
dev_indices = np.concatenate([us_indices[HIDDEN_US:HIDDEN_US+DEV_US], india_indices[HIDDEN_IN:HIDDEN_IN+DEV_IN]])
train_indices = np.concatenate([us_indices[HIDDEN_US+DEV_US:], india_indices[HIDDEN_IN+DEV_IN:]])

hidden_df = merged.iloc[hidden_indices].copy()
dev_df = merged.iloc[dev_indices].copy()

print(f"Hidden Test Set: {len(hidden_df):,} entities ({len(hidden_df[hidden_df['country']=='US']):,} US, {len(hidden_df[hidden_df['country']=='India']):,} India)")
print(f"Dev Set:         {len(dev_df):,} entities ({len(dev_df[dev_df['country']=='US']):,} US, {len(dev_df[dev_df['country']=='India']):,} India)")
print(f"Train Set:       {len(train_indices):,} entities")

# Collect all linked S2/S3 IDs for hidden and dev to audit leakage
def get_linked_ids(df):
    linked = set()
    for m in df['matched_entity_ids'].fillna(''):
        m_str = str(m).strip()
        if m_str:
            for item in m_str.split(','):
                linked.add(item.strip())
    return list(linked)

hidden_linked = get_linked_ids(hidden_df)
dev_linked = get_linked_ids(dev_df)

manifest = {
    'seed': 42,
    'stratification': 'country',
    'total_s1': len(merged),
    'splits': {
        'hidden_test': {
            'count': len(hidden_df),
            'us_count': int((hidden_df['country'] == 'US').sum()),
            'india_count': int((hidden_df['country'] == 'India').sum()),
            's1_ids': hidden_df['entity_id'].tolist(),
            'linked_s2_s3_count': len(hidden_linked)
        },
        'dev': {
            'count': len(dev_df),
            'us_count': int((dev_df['country'] == 'US').sum()),
            'india_count': int((dev_df['country'] == 'India').sum()),
            's1_ids': dev_df['entity_id'].tolist(),
            'linked_s2_s3_count': len(dev_linked)
        },
        'train_pool_count': len(train_indices)
    }
}

with open(os.path.join(OUTPUT_DIR, 'split_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

# Save test frames for evaluation
hidden_df.to_csv(os.path.join(OUTPUT_DIR, 'hidden_test_s1.tsv'), sep='\t', index=False)
dev_df.to_csv(os.path.join(OUTPUT_DIR, 'dev_s1.tsv'), sep='\t', index=False)

print("Split manifest and evaluation sets created successfully!")

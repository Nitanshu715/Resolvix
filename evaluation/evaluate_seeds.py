"""
Evaluates split stability across 5 random seeds (42, 137, 2026, 777, 9999).
Extracts the ground truth distribution, singleton ratio, entity link density,
and computes variance across splits to verify holdout stability.
"""

import os
import json
import numpy as np
import pandas as pd

OUTPUT_DIR = 'evaluation'
SEEDS = [42, 137, 2026, 777, 9999]

print("Reading train_source1.tsv and train_ground_truth.tsv for multi-seed split audit...")
s1_df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
gt_df = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')

merged = pd.merge(s1_df[['entity_id', 'country']], 
                  gt_df[['source1_entity_id', 'matched_entity_ids']], 
                  left_on='entity_id', right_on='source1_entity_id')

seed_audit = {}

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    us_indices = merged[merged['country'] == 'US'].index.values.copy()
    in_indices = merged[merged['country'] == 'India'].index.values.copy()
    
    rng.shuffle(us_indices)
    rng.shuffle(in_indices)
    
    hidden_indices = np.concatenate([us_indices[:3000], in_indices[:2000]])
    sample_df = merged.iloc[hidden_indices]
    
    # Ground truth parsing
    gts = sample_df['matched_entity_ids'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()] if str(x).strip() else [])
    num_matches = gts.apply(len)
    
    singletons = int((num_matches == 0).sum())
    singleton_rate = float(singletons / len(sample_df))
    avg_links = float(num_matches.mean())
    std_links = float(num_matches.std())
    
    seed_audit[f'Seed_{seed}'] = {
        'n_entities': len(sample_df),
        'us_entities': 3000,
        'india_entities': 2000,
        'singleton_count': singletons,
        'singleton_rate': singleton_rate,
        'mean_links_per_entity': avg_links,
        'std_links_per_entity': std_links,
        's1_id_sample': sample_df['entity_id'].iloc[:5].tolist()
    }

singleton_rates = [v['singleton_rate'] for v in seed_audit.values()]
seed_audit['stability_summary'] = {
    'mean_singleton_rate': float(np.mean(singleton_rates)),
    'std_singleton_rate': float(np.std(singleton_rates)),
    'min_singleton_rate': float(np.min(singleton_rates)),
    'max_singleton_rate': float(np.max(singleton_rates)),
}

with open(os.path.join(OUTPUT_DIR, 'seed_results.json'), 'w', encoding='utf-8') as f:
    json.dump(seed_audit, f, indent=2)

print("Multi-seed stability audit saved to evaluation/seed_results.json")

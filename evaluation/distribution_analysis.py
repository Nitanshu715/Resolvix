"""
Step 17: Distribution Analysis Across Train, Dev, Hidden Test, and Amazon Test Sets.
"""

import json
import os
import pandas as pd
from collections import Counter

print("Loading dataset metadata...")

# Train S1
train_s1_countries = Counter()
for chunk in pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', chunksize=500000, usecols=['country']):
    train_s1_countries.update(chunk['country'])

# Test S1
test_s1_countries = Counter()
for chunk in pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', chunksize=500000, usecols=['country']):
    test_s1_countries.update(chunk['country'])

# Hidden Test S1
hidden_df = pd.read_csv('evaluation/hidden_test_s1.tsv', sep='\t')
hidden_s1_countries = Counter(hidden_df['country'])

# Dev S1
dev_df = pd.read_csv('evaluation/dev_s1.tsv', sep='\t')
dev_s1_countries = Counter(dev_df['country'])

# Test predictions from matching_results.tsv
test_pred_empty_by_c = Counter()
test_pred_total_by_c = Counter()
s1_country_map = {}
for chunk in pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', chunksize=500000):
    for eid, c in zip(chunk['entity_id'], chunk['country']):
        s1_country_map[eid] = c

for chunk in pd.read_csv('matching_results.tsv', sep='\t', chunksize=500000):
    for eid, m in zip(chunk['source1_entity_id'], chunk['matched_entity_ids'].fillna('')):
        c = s1_country_map[eid]
        test_pred_total_by_c[c] += 1
        if not str(m).strip():
            test_pred_empty_by_c[c] += 1

analysis = {
    'train_s1': {
        'total': sum(train_s1_countries.values()),
        'by_country': dict(train_s1_countries),
        'proportions': {k: v / sum(train_s1_countries.values()) for k, v in train_s1_countries.items()}
    },
    'hidden_test_s1': {
        'total': len(hidden_df),
        'by_country': dict(hidden_s1_countries),
        'proportions': {k: v / len(hidden_df) for k, v in hidden_s1_countries.items()}
    },
    'dev_s1': {
        'total': len(dev_df),
        'by_country': dict(dev_s1_countries),
        'proportions': {k: v / len(dev_df) for k, v in dev_s1_countries.items()}
    },
    'amazon_test_s1': {
        'total': sum(test_s1_countries.values()),
        'by_country': dict(test_s1_countries),
        'proportions': {k: v / sum(test_s1_countries.values()) for k, v in test_s1_countries.items()}
    },
    'distribution_shift_flags': {
        'unseen_country_in_test': 'France' in test_s1_countries and 'France' not in train_s1_countries,
        'unseen_country_count': test_s1_countries.get('France', 0),
        'unseen_country_pct': test_s1_countries.get('France', 0) / sum(test_s1_countries.values()),
        'india_shift': {
            'train_pct': train_s1_countries['India'] / sum(train_s1_countries.values()),
            'test_pct': test_s1_countries['India'] / sum(test_s1_countries.values())
        }
    },
    'current_submission_diagnostics': {
        'matching_results_total_rows': sum(test_pred_total_by_c.values()),
        'matching_results_total_empty': sum(test_pred_empty_by_c.values()),
        'matching_results_empty_pct': sum(test_pred_empty_by_c.values()) / sum(test_pred_total_by_c.values()),
        'by_country_empty_pct': {c: test_pred_empty_by_c[c] / test_pred_total_by_c[c] for c in test_pred_total_by_c}
    }
}

with open('evaluation/distribution_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2)

print("Distribution analysis saved to evaluation/distribution_analysis.json")

"""
Complete Runner for Baselines A, B, C, D on Held-Out Hidden Test Set (5,000 Entities).

Baselines:
- Baseline A: All Empty Singletons
- Baseline B: Exact Normalized Name Match Only
- Baseline C: Simple Fuzzy Token Set Ratio (RapidFuzz >= 80)
- Baseline D: Current Frozen Competition Pipeline (Multi-Channel Blocking + RapidFuzz 74 + One-Owner)

Calculates exact Macro-F0.5, Precision, Recall, Singleton Accuracy, and Country Breakdowns.
Saves to evaluation/baseline_results.json.
"""

import os
import sys
import gc
import json
import time
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, 'code/business_entity_resolution/src')
sys.path.insert(0, 'evaluation')
from metric_checker import compute_macro_f05
from blocking import clean_text_for_tokenization
from predict import build_multi_channel_index, extract_multi_channel_keys, phonetic_normalize

OUTPUT_DIR = 'evaluation'
hidden_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'hidden_test_s1.tsv'), sep='\t')
print(f"Loaded {len(hidden_df):,} Hidden Test S1 entities.")

ground_truth = {}
for eid, m in zip(hidden_df['entity_id'], hidden_df['matched_entity_ids'].fillna('')):
    m_str = str(m).strip()
    ground_truth[eid] = [x.strip() for x in m_str.split(',') if x.strip()] if m_str else []

n_entities = len(hidden_df)
true_singletons = sum(1 for m in ground_truth.values() if len(m) == 0)
print(f"Ground Truth: {n_entities} entities, {true_singletons} singletons ({true_singletons/n_entities*100:.2f}%)")

baseline_results = {}

# -------------------------------------------------------------
# Baseline A: All-Singleton (Predict Nothing)
# -------------------------------------------------------------
print("\n--- Evaluating Baseline A: All Singletons ---")
preds_a = {eid: [] for eid in hidden_df['entity_id']}
res_a = compute_macro_f05(ground_truth, preds_a)
# Remove per_entity_scores for compact json
clean_res_a = {k: v for k, v in res_a.items() if k != 'per_entity_scores'}
print(f"Baseline A Macro F0.5: {res_a['macro_f05']:.4f} | Precision: {res_a['macro_precision']:.4f} | Recall: {res_a['macro_recall']:.4f} | Singleton Acc: {res_a['singleton_accuracy']:.4f}")
baseline_results['Baseline A (All Singletons)'] = clean_res_a

# Save intermediate
with open(os.path.join(OUTPUT_DIR, 'baseline_results.json'), 'w', encoding='utf-8') as f:
    json.dump(baseline_results, f, indent=2)
print("Baseline A recorded.")

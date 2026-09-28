"""
Strict Full-Corpus Retrieval & Blocking Evaluator.

Evaluates candidate generation recall against the FULL training S2 & S3 corpus
(over 5,000,000 S2 records and 5,200,000 S3 records in train).
ZERO DISTRACTOR SUBSAMPLING.

Calculates:
- Candidate Recall@1
- Candidate Recall@5
- Candidate Recall@10
- Candidate Recall@25
- % S1 with all true linked records retrievable (Entity Complete Recall)
- S2 vs S3 recall breakdown
- US vs India recall breakdown
"""

import json
import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, 'code/business_entity_resolution/src')
from predict import build_multi_channel_index, extract_multi_channel_keys

OUTPUT_DIR = 'evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting Full-Corpus Retrieval & Blocking Evaluation...")

# Load hidden test S1
hidden_df = pd.read_csv('evaluation/hidden_test_s1.tsv', sep='\t')
print(f"Loaded {len(hidden_df):,} held-out Hidden Test S1 entities.")

# Parse ground truth
gt_matches = {}
for eid, m in zip(hidden_df['entity_id'], hidden_df['matched_entity_ids'].fillna('')):
    m_str = str(m).strip()
    gt_matches[eid] = set(m_str.split(',')) if m_str else set()

results_by_country = {}

for country in ['US', 'India']:
    print(f"\n{'='*60}\nEvaluating Retrieval for Country: {country}\n{'='*60}")
    c_s1 = hidden_df[hidden_df['country'] == country].reset_index(drop=True)
    n_s1 = len(c_s1)

    # Load FULL S2 for this country from train
    print(f"Loading full train_source2 for {country}...")
    s2_chunks = []
    for chunk in pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
        f = chunk[chunk['country'] == country]
        if len(f):
            s2_chunks.append(f)
    s2_df = pd.concat(s2_chunks, ignore_index=True)
    del s2_chunks
    print(f"Loaded {len(s2_df):,} S2 records for {country}.")

    # Load FULL S3 for this country from train
    print(f"Loading full train_source3 for {country}...")
    s3_chunks = []
    for chunk in pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
        f = chunk[chunk['country'] == country]
        if len(f):
            s3_chunks.append(f)
    s3_df = pd.concat(s3_chunks, ignore_index=True)
    del s3_chunks
    print(f"Loaded {len(s3_df):,} S3 records for {country}.")

    # Build Multi-Channel indices
    print("Building S2 index...")
    s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp = build_multi_channel_index(
        s2_df['business_name'].tolist(), s2_df['business_address'].tolist()
    )
    s2_eids = s2_df['entity_id'].values
    del s2_df

    print("Building S3 index...")
    s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp = build_multi_channel_index(
        s3_df['business_name'].tolist(), s3_df['business_address'].tolist()
    )
    s3_eids = s3_df['entity_id'].values
    del s3_df

    # Query loop
    k_values = [1, 5, 10, 25]
    recall_at_k = {k: [] for k in k_values}
    s2_recall_at_k = {k: [] for k in k_values}
    s3_recall_at_k = {k: [] for k in k_values}
    complete_recall = []

    s1_names = c_s1['business_name'].tolist()
    s1_addrs = c_s1['business_address'].tolist()
    s1_ids = c_s1['entity_id'].tolist()

    t0 = time.time()
    for i in range(n_s1):
        s1_id = s1_ids[i]
        true_all = gt_matches[s1_id]
        true_s2 = {x for x in true_all if x.startswith('S2-')}
        true_s3 = {x for x in true_all if x.startswith('S3-')}

        if not true_all:
            continue  # singletons have no retrieval target to evaluate recall against

        nt, nakeys, comp = extract_multi_channel_keys(s1_names[i], s1_addrs[i])

        # Retrieve S2
        scores_s2 = {}
        if comp in s2_comp_p:
            for d in s2_comp_p[comp]:
                scores_s2[d] = 30.0
        for nak in nakeys:
            if nak in s2_numaddr_idx:
                for d in s2_numaddr_idx[nak][:50]:
                    scores_s2[d] = scores_s2.get(d, 0.0) + 15.0
        for w in nt:
            if w in s2_name_idx:
                docs, idf = s2_name_idx[w]
                for d in docs[:100]:
                    scores_s2[d] = scores_s2.get(d, 0.0) + idf
        top_s2 = sorted(scores_s2, key=scores_s2.get, reverse=True)[:25] if scores_s2 else []

        # Retrieve S3
        scores_s3 = {}
        if comp in s3_comp_p:
            for d in s3_comp_p[comp]:
                scores_s3[d] = 30.0
        for nak in nakeys:
            if nak in s3_numaddr_idx:
                for d in s3_numaddr_idx[nak][:50]:
                    scores_s3[d] = scores_s3.get(d, 0.0) + 15.0
        for w in nt:
            if w in s3_name_idx:
                docs, idf = s3_name_idx[w]
                for d in docs[:100]:
                    scores_s3[d] = scores_s3.get(d, 0.0) + idf
        top_s3 = sorted(scores_s3, key=scores_s3.get, reverse=True)[:25] if scores_s3 else []

        retrieved_s2 = [s2_eids[d] for d in top_s2]
        retrieved_s3 = [s3_eids[d] for d in top_s3]

        for k in k_values:
            cands_k = set(retrieved_s2[:k] + retrieved_s3[:k])
            hits = len(cands_k & true_all)
            recall_at_k[k].append(hits / len(true_all))

            if true_s2:
                s2_hits = len(set(retrieved_s2[:k]) & true_s2)
                s2_recall_at_k[k].append(s2_hits / len(true_s2))
            if true_s3:
                s3_hits = len(set(retrieved_s3[:k]) & true_s3)
                s3_recall_at_k[k].append(s3_hits / len(true_s3))

        cands_all = set(retrieved_s2 + retrieved_s3)
        complete_recall.append(1.0 if true_all.issubset(cands_all) else 0.0)

    results_by_country[country] = {
        'n_eval_queries': len(complete_recall),
        'complete_recall': float(np.mean(complete_recall)),
        'recall_at_k': {f'Recall@{k}': float(np.mean(recall_at_k[k])) for k in k_values},
        's2_recall_at_k': {f'S2_Recall@{k}': float(np.mean(s2_recall_at_k[k])) for k in k_values},
        's3_recall_at_k': {f'S3_Recall@{k}': float(np.mean(s3_recall_at_k[k])) for k in k_values}
    }
    print(f"Results for {country}:")
    print(f"  Complete Recall: {results_by_country[country]['complete_recall']*100:.2f}%")
    for k in k_values:
        print(f"  Recall@{k}: {results_by_country[country]['recall_at_k'][f'Recall@{k}']*100:.2f}%")

    del s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp, s2_eids
    del s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp, s3_eids

with open(os.path.join(OUTPUT_DIR, 'retrieval_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results_by_country, f, indent=2)

print("\nRetrieval results saved to evaluation/retrieval_results.json")

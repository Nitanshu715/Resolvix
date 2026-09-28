"""
Clean, Memory-Disciplined Pipeline and Baseline Evaluator.
Evaluates:
- Baseline A: All Singletons
- Baseline B: Exact Normalized Name Match
- Baseline C: RapidFuzz >= 80
- Baseline D: Frozen Current Pipeline (Multi-Channel + RapidFuzz 74 + One-Owner)

Also computes:
- Country Breakdown (US vs India)
- 1,000 Bootstrap 95% Confidence Intervals
- Adversarial Slice Stress Tests

Saves all metrics strictly to JSON artifacts.
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

def main():
    print("Loading Held-Out Hidden Test Set (5,000 S1 entities)...")
    hidden_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'hidden_test_s1.tsv'), sep='\t')
    n_entities = len(hidden_df)

    ground_truth = {}
    for eid, m in zip(hidden_df['entity_id'], hidden_df['matched_entity_ids'].fillna('')):
        m_str = str(m).strip()
        ground_truth[eid] = [x.strip() for x in m_str.split(',') if x.strip()] if m_str else []

    true_singletons = sum(1 for m in ground_truth.values() if len(m) == 0)
    print(f"Ground Truth: {n_entities} entities, {true_singletons} singletons ({true_singletons/n_entities*100:.2f}%)")

    baseline_results = {}

    # 1. Baseline A: All Singletons
    preds_a = {eid: [] for eid in hidden_df['entity_id']}
    res_a = compute_macro_f05(ground_truth, preds_a)
    baseline_results['Baseline A (All Singletons)'] = {k: v for k, v in res_a.items() if k != 'per_entity_scores'}
    print(f"Baseline A Macro F0.5: {res_a['macro_f05']:.4f}")

    pipeline_preds = {}
    baseline_b_preds = {}
    baseline_c_preds = {}

    countries = ['US', 'India']

    for country in countries:
        print(f"\n================ Loading & Indexing {country} ================")
        c_hidden = hidden_df[hidden_df['country'] == country].reset_index(drop=True)
        c_n = len(c_hidden)

        # Stream load S2
        s2_chunks = []
        for chunk in pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
            f = chunk[chunk['country'] == country]
            if len(f):
                s2_chunks.append(f)
        s2_df = pd.concat(s2_chunks, ignore_index=True)
        del s2_chunks
        print(f"Loaded {len(s2_df):,} S2 records for {country}.")

        # Stream load S3
        s3_chunks = []
        for chunk in pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
            f = chunk[chunk['country'] == country]
            if len(f):
                s3_chunks.append(f)
        s3_df = pd.concat(s3_chunks, ignore_index=True)
        del s3_chunks
        print(f"Loaded {len(s3_df):,} S3 records for {country}.")

        s2_eids = s2_df['entity_id'].values
        s2_names = s2_df['business_name'].tolist()
        s2_addrs = s2_df['business_address'].tolist()
        del s2_df
        gc.collect()

        s3_eids = s3_df['entity_id'].values
        s3_names = s3_df['business_name'].tolist()
        s3_addrs = s3_df['business_address'].tolist()
        del s3_df
        gc.collect()

        print("Building S2 index...")
        s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp = build_multi_channel_index(s2_names, s2_addrs)
        del s2_names, s2_addrs
        gc.collect()

        print("Building S3 index...")
        s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp = build_multi_channel_index(s3_names, s3_addrs)
        del s3_names, s3_addrs
        gc.collect()

        c_eids = c_hidden['entity_id'].values
        c_names = c_hidden['business_name'].tolist()
        c_addrs = c_hidden['business_address'].tolist()

        target_to_s1 = defaultdict(list)
        s1_raw_matches = defaultdict(list)

        print(f"Scoring {c_n} queries for {country}...")
        MATCH_THRESHOLD = 74

        for i in range(c_n):
            s1_id = c_eids[i]
            s1_n = c_names[i]
            s1_a = c_addrs[i]

            nt, nakeys, comp = extract_multi_channel_keys(s1_n, s1_a)

            # Query S2
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
            top_s2 = sorted(scores_s2, key=scores_s2.get, reverse=True)[:30] if scores_s2 else []

            # Query S3
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
            top_s3 = sorted(scores_s3, key=scores_s3.get, reverse=True)[:30] if scores_s3 else []

            s1_full_str = phonetic_normalize(clean_text_for_tokenization(str(s1_n) + ' ' + str(s1_a)))
            s1_comp_name = ''.join(re.sub(r'[^a-z0-9]', ' ', s1_full_str).split())
            norm_s1_pure_name = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]', ' ', str(s1_n).lower())).strip()

            # Baseline B: Exact Normalized Name Matches within candidate set
            b_matches = []
            for d in top_s2:
                # s2_full contains normalized name + address; check prefix name equality
                cand_pure = s2_full[d].split()[0] if s2_full[d] else ''
                if norm_s1_pure_name and norm_s1_pure_name == s2_full[d][:len(norm_s1_pure_name)]:
                    b_matches.append(s2_eids[d])
            for d in top_s3:
                if norm_s1_pure_name and norm_s1_pure_name == s3_full[d][:len(norm_s1_pure_name)]:
                    b_matches.append(s3_eids[d])
            baseline_b_preds[s1_id] = b_matches

            # Baseline C: Simple Fuzzy >= 80
            c_matches = []
            for d in top_s2:
                sim = fuzz.token_set_ratio(s1_full_str, s2_full[d])
                if sim >= 80:
                    c_matches.append(s2_eids[d])
            for d in top_s3:
                sim = fuzz.token_set_ratio(s1_full_str, s3_full[d])
                if sim >= 80:
                    c_matches.append(s3_eids[d])
            baseline_c_preds[s1_id] = c_matches

            # Baseline D (Frozen Pipeline): RapidFuzz >= 74 + Domain Boost + One-Owner
            for d in top_s2:
                sim = fuzz.token_set_ratio(s1_full_str, s2_full[d])
                t_comp = s2_comp[d]
                if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                    sim = max(sim, 85)
                if sim >= MATCH_THRESHOLD:
                    teid = s2_eids[d]
                    s1_raw_matches[s1_id].append((teid, sim))
                    target_to_s1[teid].append((s1_id, sim))

            for d in top_s3:
                sim = fuzz.token_set_ratio(s1_full_str, s3_full[d])
                t_comp = s3_comp[d]
                if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                    sim = max(sim, 85)
                if sim >= MATCH_THRESHOLD:
                    teid = s3_eids[d]
                    s1_raw_matches[s1_id].append((teid, sim))
                    target_to_s1[teid].append((s1_id, sim))

        # One-owner resolution for Baseline D
        owner_map = {}
        for target_eid, s1_cands in target_to_s1.items():
            best_s1 = max(s1_cands, key=lambda x: x[1])[0]
            owner_map[target_eid] = best_s1

        for s1_id in c_eids:
            raw = s1_raw_matches.get(s1_id, [])
            final = [teid for teid, _ in raw if owner_map.get(teid) == s1_id]
            pipeline_preds[s1_id] = final

        del s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp, s2_eids
        del s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp, s3_eids
        del target_to_s1, s1_raw_matches, owner_map
        gc.collect()

    print("\n--- Saving intermediate predictions ---")
    with open(os.path.join(OUTPUT_DIR, 'intermediate_preds.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'pipeline_preds': pipeline_preds,
            'baseline_b_preds': baseline_b_preds,
            'baseline_c_preds': baseline_c_preds
        }, f)

    print("\n--- Computing Baseline Evaluations on Full Hidden Test Set ---")
    res_b = compute_macro_f05(ground_truth, baseline_b_preds)
    baseline_results['Baseline B (Exact Normalized Name)'] = {k: v for k, v in res_b.items() if k != 'per_entity_scores'}
    print(f"Baseline B Macro F0.5: {res_b['macro_f05']:.4f} | Prec: {res_b['macro_precision']:.4f} | Rec: {res_b['macro_recall']:.4f} | SingAcc: {res_b['singleton_accuracy']:.4f}")

    res_c = compute_macro_f05(ground_truth, baseline_c_preds)
    baseline_results['Baseline C (Simple Fuzzy >= 80)'] = {k: v for k, v in res_c.items() if k != 'per_entity_scores'}
    print(f"Baseline C Macro F0.5: {res_c['macro_f05']:.4f} | Prec: {res_c['macro_precision']:.4f} | Rec: {res_c['macro_recall']:.4f} | SingAcc: {res_c['singleton_accuracy']:.4f}")

    res_d = compute_macro_f05(ground_truth, pipeline_preds)
    baseline_results['Baseline D (Current Frozen Pipeline)'] = {k: v for k, v in res_d.items() if k != 'per_entity_scores'}
    print(f"Baseline D Macro F0.5: {res_d['macro_f05']:.4f} | Prec: {res_d['macro_precision']:.4f} | Rec: {res_d['macro_recall']:.4f} | SingAcc: {res_d['singleton_accuracy']:.4f}")

    with open(os.path.join(OUTPUT_DIR, 'baseline_results.json'), 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2)

    # Country Breakdown for Baseline D
    us_gt = {eid: ground_truth[eid] for eid in hidden_df[hidden_df['country'] == 'US']['entity_id']}
    us_pred = {eid: pipeline_preds[eid] for eid in us_gt}
    us_res = compute_macro_f05(us_gt, us_pred)

    in_gt = {eid: ground_truth[eid] for eid in hidden_df[hidden_df['country'] == 'India']['entity_id']}
    in_pred = {eid: pipeline_preds[eid] for eid in in_gt}
    in_res = compute_macro_f05(in_gt, in_pred)

    hidden_test_results = {
        'overall': {k: v for k, v in res_d.items() if k != 'per_entity_scores'},
        'US': {k: v for k, v in us_res.items() if k != 'per_entity_scores'},
        'India': {k: v for k, v in in_res.items() if k != 'per_entity_scores'},
    }
    with open(os.path.join(OUTPUT_DIR, 'hidden_test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(hidden_test_results, f, indent=2)

    # 1,000 Bootstrap 95% Confidence Intervals
    print("\n--- Running 1,000 Bootstrap Resamples ---")
    per_entity_scores = list(res_d['per_entity_scores'].values())
    rng = np.random.RandomState(42)
    boot_means = []
    n_samples = len(per_entity_scores)
    for _ in range(1000):
        sample = rng.choice(per_entity_scores, size=n_samples, replace=True)
        boot_means.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    ci_mean = float(np.mean(boot_means))
    ci_std = float(np.std(boot_means))

    bootstrap_results = {
        'n_resamples': 1000,
        'bootstrap_mean': ci_mean,
        'bootstrap_std': ci_std,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'point_estimate': res_d['macro_f05']
    }
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    with open(os.path.join(OUTPUT_DIR, 'bootstrap_results.json'), 'w', encoding='utf-8') as f:
        json.dump(bootstrap_results, f, indent=2)

    # Stress Tests
    print("\n--- Running Adversarial Stress Tests ---")
    singleton_ids = [eid for eid, g in ground_truth.items() if len(g) == 0]
    sing_gt = {eid: ground_truth[eid] for eid in singleton_ids}
    sing_pred = {eid: pipeline_preds[eid] for eid in singleton_ids}
    res_stress_sing = compute_macro_f05(sing_gt, sing_pred)

    dense_ids = [eid for eid, g in ground_truth.items() if len(g) >= 3]
    dense_gt = {eid: ground_truth[eid] for eid in dense_ids}
    dense_pred = {eid: pipeline_preds[eid] for eid in dense_ids}
    res_stress_dense = compute_macro_f05(dense_gt, dense_pred)

    single_ids = [eid for eid, g in ground_truth.items() if len(g) == 1]
    single_gt = {eid: ground_truth[eid] for eid in single_ids}
    single_pred = {eid: pipeline_preds[eid] for eid in single_ids}
    res_stress_single = compute_macro_f05(single_gt, single_pred)

    hidden_df_indexed = hidden_df.set_index('entity_id')
    short_ids = [eid for eid in ground_truth if len(str(hidden_df_indexed.loc[eid, 'business_name']).split()) <= 2]
    short_gt = {eid: ground_truth[eid] for eid in short_ids}
    short_pred = {eid: pipeline_preds[eid] for eid in short_ids}
    res_stress_short = compute_macro_f05(short_gt, short_pred)

    stress_results = {
        'pure_singletons': {k: v for k, v in res_stress_sing.items() if k != 'per_entity_scores'},
        'dense_entities_ge3_matches': {k: v for k, v in res_stress_dense.items() if k != 'per_entity_scores'},
        'single_match_entities': {k: v for k, v in res_stress_single.items() if k != 'per_entity_scores'},
        'short_names_le2_words': {k: v for k, v in res_stress_short.items() if k != 'per_entity_scores'}
    }
    with open(os.path.join(OUTPUT_DIR, 'stress_test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(stress_results, f, indent=2)

    print("\nSUCCESS: All baselines and stress tests finished and saved.")

if __name__ == '__main__':
    main()

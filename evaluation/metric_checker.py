"""
Step 1 & Step 15: Exact Amazon Metric Implementation & Sanity Check Suite.

Competition Metric:
Macro F0.5 per Source-1 entity.
For each S1 entity:
  Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
  Recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
  F0.5      = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0

Special Singleton Rule:
  If Truth is empty (singleton):
    - Predicted empty => F0.5 = 1.0 (exact correct singleton identification)
    - Predicted non-empty => F0.5 = 0.0 (false positive links on singleton)
  If Truth is non-empty:
    - Predicted empty => F0.5 = 0.0 (missed all links)

Macro-Average:
  Mean of F0.5 across all S1 entities in the evaluation set.
"""

import json
import numpy as np


def score_entity_f05_v1(pred_set: set, truth_set: set) -> float:
    """Implementation 1: Direct definition."""
    if not pred_set and not truth_set:
        return 1.0
    if not truth_set or not pred_set:
        return 0.0

    tp = len(pred_set & truth_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(truth_set)

    beta_sq = 0.25  # 0.5^2
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def score_entity_f05_v2(pred_set: set, truth_set: set) -> float:
    """Implementation 2: Independent algebraic reformulation in terms of TP, FP, FN."""
    # TP: pred & truth
    # FP: pred - truth
    # FN: truth - pred
    if not pred_set and not truth_set:
        return 1.0
    if not truth_set or not pred_set:
        return 0.0

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    # F_beta = (1 + beta^2)*TP / ((1 + beta^2)*TP + FP + beta^2*FN)
    # With F0.5 = (1.25 * P * R) / (0.25 * P + R):
    # P = TP/(TP+FP), R = TP/(TP+FN)
    # F0.5 = (1.25 * TP / ((TP+FP)(TP+FN))) / ((0.25*TP*(TP+FN) + TP*(TP+FP)) / ((TP+FP)(TP+FN)))
    #      = 1.25 * TP / (0.25*(TP+FN) + (TP+FP))
    #      = 1.25 * TP / (1.25*TP + FP + 0.25*FN)
    #      = 5 * TP / (5*TP + 4*FP + FN)
    denom = 5.0 * tp + 4.0 * float(fp) + float(fn)
    if denom == 0.0:
        return 0.0
    return (5.0 * tp) / denom


def compute_macro_f05(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute Macro F0.5 across all S1 entities in ground_truth.
    Returns detailed summary statistics.
    """
    scores_v1 = []
    scores_v2 = []
    precisions = []
    recalls = []

    tp_total = 0
    fp_total = 0
    fn_total = 0
    singleton_total = 0
    singleton_correct = 0

    buckets = {
        '1.0': 0,
        '0.75-1.0': 0,
        '0.50-0.75': 0,
        '0.25-0.50': 0,
        '0.0-0.25': 0,
        '0.0': 0
    }

    for s1_id, truth_set in ground_truth.items():
        truth_set = set(truth_set) if isinstance(truth_set, (list, tuple, set)) else set()
        pred_set = predictions.get(s1_id, set())
        pred_set = set(pred_set) if isinstance(pred_set, (list, tuple, set)) else set()

        s1 = score_entity_f05_v1(pred_set, truth_set)
        s2 = score_entity_f05_v2(pred_set, truth_set)
        assert abs(s1 - s2) < 1e-12, f"Discrepancy for {s1_id}: v1={s1} vs v2={s2}"

        scores_v1.append(s1)
        scores_v2.append(s2)

        # Record bucket
        if s1 == 1.0:
            buckets['1.0'] += 1
        elif s1 >= 0.75:
            buckets['0.75-1.0'] += 1
        elif s1 >= 0.50:
            buckets['0.50-0.75'] += 1
        elif s1 >= 0.25:
            buckets['0.25-0.50'] += 1
        elif s1 > 0.0:
            buckets['0.0-0.25'] += 1
        else:
            buckets['0.0'] += 1

        # Track singleton
        if not truth_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1

        # Track precision / recall per entity for reporting
        if pred_set and truth_set:
            tp = len(pred_set & truth_set)
            precisions.append(tp / len(pred_set))
            recalls.append(tp / len(truth_set))
            tp_total += tp
            fp_total += len(pred_set - truth_set)
            fn_total += len(truth_set - pred_set)
        elif not truth_set:
            fp_total += len(pred_set)
            if pred_set:
                precisions.append(0.0)
            else:
                precisions.append(1.0)
            recalls.append(1.0)
        else:  # truth_set and not pred_set
            fn_total += len(truth_set)
            precisions.append(0.0)
            recalls.append(0.0)

    n = len(scores_v1)
    return {
        'n_entities': n,
        'macro_f05': float(np.mean(scores_v1)) if n else 0.0,
        'macro_precision': float(np.mean(precisions)) if n else 0.0,
        'macro_recall': float(np.mean(recalls)) if n else 0.0,
        'singleton_accuracy': float(singleton_correct / singleton_total) if singleton_total else 1.0,
        'singleton_total': singleton_total,
        'singleton_correct': singleton_correct,
        'tp_links': tp_total,
        'fp_links': fp_total,
        'fn_links': fn_total,
        'score_distribution': {k: v / n if n else 0.0 for k, v in buckets.items()},
        'score_counts': buckets,
        'per_entity_scores': dict(zip(ground_truth.keys(), scores_v1))
    }


def run_sanity_tests():
    """Execute rigorous unit tests on metric corner cases."""
    tests = [
        {
            'name': 'Perfect match (single)',
            'pred': {'S2-1'},
            'truth': {'S2-1'},
            'expected_f05': 1.0
        },
        {
            'name': 'Perfect match (multiple)',
            'pred': {'S2-1', 'S3-2'},
            'truth': {'S2-1', 'S3-2'},
            'expected_f05': 1.0
        },
        {
            'name': 'Correct singleton (both empty)',
            'pred': set(),
            'truth': set(),
            'expected_f05': 1.0
        },
        {
            'name': 'False positive singleton (truth empty, predicted non-empty)',
            'pred': {'S2-1'},
            'truth': set(),
            'expected_f05': 0.0
        },
        {
            'name': 'False negative (truth non-empty, predicted empty)',
            'pred': set(),
            'truth': {'S2-1'},
            'expected_f05': 0.0
        },
        {
            'name': 'Partial match: 1 TP, 1 FP (truth: {S2-1}, pred: {S2-1, S2-2})',
            # TP=1, FP=1, FN=0 -> P=0.5, R=1.0 -> F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5555555556
            'pred': {'S2-1', 'S2-2'},
            'truth': {'S2-1'},
            'expected_f05': 5.0 / 9.0
        },
        {
            'name': 'Partial match: 1 TP, 1 FN (truth: {S2-1, S3-1}, pred: {S2-1})',
            # TP=1, FP=0, FN=1 -> P=1.0, R=0.5 -> F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333333333
            'pred': {'S2-1'},
            'truth': {'S2-1', 'S3-1'},
            'expected_f05': 5.0 / 6.0
        },
        {
            'name': 'Partial match: 2 TP, 1 FP, 1 FN (truth: {A, B, C}, pred: {A, B, D})',
            # TP=2, FP=1, FN=1 -> P=2/3, R=2/3 -> F0.5 = 5*2 / (5*2 + 1 + 4*1) = 10 / 15 = 2/3 = 0.6666666667
            'pred': {'A', 'B', 'D'},
            'truth': {'A', 'B', 'C'},
            'expected_f05': 2.0 / 3.0
        }
    ]

    results = []
    all_passed = True
    for t in tests:
        v1 = score_entity_f05_v1(t['pred'], t['truth'])
        v2 = score_entity_f05_v2(t['pred'], t['truth'])
        exp = t['expected_f05']
        pass_v1 = abs(v1 - exp) < 1e-9
        pass_v2 = abs(v2 - exp) < 1e-9
        passed = pass_v1 and pass_v2
        if not passed:
            all_passed = False
        results.append({
            'test_name': t['name'],
            'v1_score': v1,
            'v2_score': v2,
            'expected': exp,
            'passed': passed
        })

    return all_passed, results


if __name__ == '__main__':
    all_passed, results = run_sanity_tests()
    print(f"Sanity Check Suite Passed: {all_passed}")
    for r in results:
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  [{status}] {r['test_name']}: score={r['v1_score']:.6f} expected={r['expected']:.6f}")

    with open('evaluation/metric_tests.json', 'w', encoding='utf-8') as f:
        json.dump({'all_passed': all_passed, 'tests': results}, f, indent=2)

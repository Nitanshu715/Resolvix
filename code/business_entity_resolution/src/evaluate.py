"""
Evaluation utilities: F_0.5 macro-average score and blocking recall.

F_0.5 is precision-heavy (weights precision 4x over recall):
    F_0.5 = 1.25 * P * R / (0.25 * P + R)

Computed per-entity then macro-averaged over all S1 entities.
Singletons (true match = empty) score 1.0 when predicted empty, 0.0 otherwise.
"""

import numpy as np
import pandas as pd


def f_half(precision: float, recall: float) -> float:
    """Compute F_0.5 given precision and recall."""
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return 1.25 * precision * recall / denom


def score_entity(predicted: set, truth: set) -> float:
    """
    Compute F_0.5 for a single S1 entity.

    Args:
        predicted: set of predicted matching IDs (empty = singleton prediction)
        truth:     set of ground-truth matching IDs (empty = true singleton)

    Returns:
        F_0.5 in [0, 1]
    """
    if not predicted and not truth:
        return 1.0   # correct singleton prediction

    if not truth:
        # True singleton but we predicted some matches → precision=0
        return 0.0

    if not predicted:
        # Had true matches but predicted nothing → recall=0
        return 0.0

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_half(precision, recall)


def evaluate(
    predictions: dict,
    ground_truth_df: pd.DataFrame,
) -> float:
    """
    Compute macro-average F_0.5 across all S1 entities.

    Args:
        predictions:      dict  s1_entity_id → list of predicted matching IDs
        ground_truth_df:  DataFrame with columns
                          [source1_entity_id, matched_entity_ids]

    Returns:
        macro-average F_0.5
    """
    scores: list[float] = []

    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']

        if pd.isna(matched_str) or str(matched_str).strip() == '':
            truth: set[str] = set()
        else:
            truth = set(str(matched_str).split(','))

        pred_list = predictions.get(s1_id, [])
        pred: set[str] = set(pred_list) if pred_list else set()

        scores.append(score_entity(pred, truth))

    return float(np.mean(scores)) if scores else 0.0


def compute_blocking_metrics(
    candidates: dict,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """
    Compute recall and reduction metrics for the blocking step.

    Args:
        candidates:       dict  s1_entity_id → list of candidate IDs
        ground_truth_df:  ground-truth DataFrame

    Returns:
        dict with keys:
          blocking_recall      – fraction of true matches that appear in candidates
          avg_candidates       – mean candidate-list length per S1 entity
          reduction_ratio      – 1 − avg_candidates / total_s23_size
                                 (not computed here since we don't know total_s23)
    """
    total_matches = 0
    recalled = 0
    total_cands = 0

    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']

        if pd.isna(matched_str) or str(matched_str).strip() == '':
            truth: set[str] = set()
        else:
            truth = set(str(matched_str).split(','))

        cands: set[str] = set(candidates.get(s1_id, []))
        total_matches += len(truth)
        recalled += len(truth & cands)
        total_cands += len(cands)

    n_entities = len(ground_truth_df)
    recall = recalled / total_matches if total_matches else 1.0
    avg_cands = total_cands / n_entities if n_entities else 0.0

    return {
        'blocking_recall': recall,
        'avg_candidates': avg_cands,
        'total_pairs': total_cands,
    }

"""
Enhanced Training Script with Hard Negatives Mining, Rich 35-Features, and Margin-Aware Scoring.
"""

import os
import sys
import pickle
import logging
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))

from blocking import BlockingEngine
from features import compute_features_batch, FEATURE_NAMES, N_FEATURES
from evaluate import evaluate, compute_blocking_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SRC_DIR)
REPO_DIR = os.path.dirname(CODE_DIR)
ROOT_DIR = os.path.dirname(REPO_DIR)
STUDENT_DIR = os.path.join(ROOT_DIR, 'student_resource')
TRAIN_DIR = os.path.join(STUDENT_DIR, 'dataset', 'train')
MODEL_DIR = os.path.join(CODE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_train_data():
    logger.info("Loading training data…")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t')
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')
    logger.info(
        f"S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}"
    )
    return s1, s2, s3, gt


def make_entity_lookup(df: pd.DataFrame) -> dict:
    return {
        r.entity_id: (r.business_name, r.business_address, r.country)
        for r in df.itertuples(index=False)
    }


def make_gt_lookup(gt_df: pd.DataFrame) -> dict:
    lookup = {}
    for r in gt_df.itertuples(index=False):
        s = r.matched_entity_ids
        if pd.isna(s) or str(s).strip() == '':
            lookup[r.source1_entity_id] = frozenset()
        else:
            lookup[r.source1_entity_id] = frozenset(str(s).split(','))
    return lookup


def create_pairs_with_hard_negatives(
    candidates: dict, gt_lookup: dict, neg_ratio: int = 4, seed: int = 42
) -> pd.DataFrame:
    """
    Selects top candidates not in ground truth as HARD negatives
    instead of purely random choices.
    """
    rng = np.random.default_rng(seed)
    pos_rows, neg_rows = [], []

    for s1_id, cands in candidates.items():
        truth = gt_lookup.get(s1_id, frozenset())
        pos = [c for c in cands if c in truth]
        # Candidates that ranked highest in blocking but are NOT in GT are hard negatives
        hard_negs = [c for c in cands if c not in truth]

        for c in pos:
            pos_rows.append((s1_id, c, 1))

        # Take hard negatives (top retrieved)
        n_neg = min(len(hard_negs), max(len(pos) * neg_ratio, 3))
        for c in hard_negs[:n_neg]:
            neg_rows.append((s1_id, c, 0))

    rows = pos_rows + neg_rows
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id', 'label'])
    n_pos = (df['label'] == 1).sum()
    n_neg = (df['label'] == 0).sum()
    logger.info(f"  Training pairs: {n_pos:,} positive, {n_neg:,} hard-negative")
    return df


def train_lgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    params = dict(
        objective='binary',
        metric='binary_logloss',
        n_estimators=800,
        learning_rate=0.06,
        max_depth=7,
        num_leaves=63,
        min_child_samples=25,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=4,
        lambda_l1=0.2,
        lambda_l2=0.2,
        verbose=-1,
        n_jobs=-1,
    )
    model = lgb.LGBMClassifier(**params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=50),
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    return model


def optimise_threshold(model, X_val, val_pairs_df, s1_val_eids, gt_lookup_val):
    probs = model.predict_proba(X_val)[:, 1]
    val_pairs_df = val_pairs_df.copy()
    val_pairs_df['prob'] = probs

    best_score = -1.0
    best_thr = 0.5

    val_gt_rows = [
        {
            'source1_entity_id': eid,
            'matched_entity_ids': ','.join(gt_lookup_val.get(eid, frozenset())),
        }
        for eid in s1_val_eids
    ]
    val_gt_df = pd.DataFrame(val_gt_rows)

    for thr in np.arange(0.20, 0.90, 0.05):
        preds = {eid: [] for eid in s1_val_eids}
        for s1_id, grp in val_pairs_df.groupby('source1_entity_id'):
            matched = grp.loc[grp['prob'] >= thr, 'candidate_entity_id'].tolist()
            preds[s1_id] = matched

        score = evaluate(preds, val_gt_df)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    logger.info(f"  Best threshold={best_thr:.2f}  val F_0.5={best_score:.4f}")
    return best_thr, best_score


def main():
    t_start = time.time()

    s1, s2, s3, gt = load_train_data()

    logger.info("Sampling 40,000 S1 records stratified across countries for model training...")
    sample_s1 = s1.groupby('country', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 20000), random_state=42)
    ).reset_index(drop=True)

    s1_ids = sample_s1['entity_id'].values.copy()
    rng = np.random.default_rng(42)
    rng.shuffle(s1_ids)
    n_train = int(len(s1_ids) * 0.8)
    train_ids = frozenset(s1_ids[:n_train])
    val_ids = frozenset(s1_ids[n_train:])

    s1_train = sample_s1[sample_s1['entity_id'].isin(train_ids)].reset_index(drop=True)
    s1_val = sample_s1[sample_s1['entity_id'].isin(val_ids)].reset_index(drop=True)

    s1_lookup = make_entity_lookup(sample_s1)
    s23_df = pd.concat([s2, s3], ignore_index=True)
    s23_lookup = make_entity_lookup(s23_df)
    gt_lookup = make_gt_lookup(gt)

    engine = BlockingEngine(top_k=50)

    # ── Train split blocking ───────────────────────────────────────────────────
    logger.info("\n== Multi-Pass Blocking: TRAIN sample ==")
    train_cands = engine.generate_candidates(s1_train, s2, s3)

    train_gt_df = gt[gt['source1_entity_id'].isin(train_ids)]
    bm = compute_blocking_metrics(train_cands, train_gt_df)
    logger.info(
        f"  Blocking recall={bm['blocking_recall']:.4f}  "
        f"avg_cands={bm['avg_candidates']:.1f}  "
        f"total_pairs={bm['total_pairs']:,}"
    )

    # ── Create & featurise training pairs with HARD NEGATIVES ─────────────────
    logger.info("\n== Mining Hard Negatives & Creating Training Pairs ==")
    gt_lookup_train = {k: gt_lookup[k] for k in train_ids if k in gt_lookup}
    train_pairs = create_pairs_with_hard_negatives(train_cands, gt_lookup_train, neg_ratio=4)

    logger.info(f"Computing 35 features for {len(train_pairs):,} pairs…")
    X_train = compute_features_batch(train_pairs, s1_lookup, s23_lookup)
    y_train = train_pairs['label'].values

    # ── Val split blocking ─────────────────────────────────────────────────────
    logger.info("\n== Multi-Pass Blocking: VAL sample ==")
    val_cands = engine.generate_candidates(s1_val, s2, s3)

    val_gt_df = gt[gt['source1_entity_id'].isin(val_ids)]
    bm_val = compute_blocking_metrics(val_cands, val_gt_df)
    logger.info(
        f"  Val blocking recall={bm_val['blocking_recall']:.4f}  "
        f"avg_cands={bm_val['avg_candidates']:.1f}"
    )

    gt_lookup_val = {k: gt_lookup[k] for k in val_ids if k in gt_lookup}

    val_rows = []
    for s1_id, cands in val_cands.items():
        truth = gt_lookup_val.get(s1_id, frozenset())
        for c in cands:
            val_rows.append((s1_id, c, int(c in truth)))
    val_pairs = pd.DataFrame(
        val_rows, columns=['source1_entity_id', 'candidate_entity_id', 'label']
    )

    logger.info(f"Computing 35 features for {len(val_pairs):,} val pairs…")
    X_val = compute_features_batch(val_pairs, s1_lookup, s23_lookup)
    y_val = val_pairs['label'].values

    # ── Train model ────────────────────────────────────────────────────────────
    logger.info("\n== Training Enhanced LightGBM Classifier ==")
    model = train_lgbm(X_train, y_train, X_val, y_val)

    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    logger.info("Top-12 feature importances:")
    for name, score in imp[:12]:
        logger.info(f"  {name:30s} {score:.0f}")

    # ── Optimise threshold ─────────────────────────────────────────────────────
    logger.info("\n== Optimising threshold on val set for F_0.5 ==")
    threshold, val_f05 = optimise_threshold(
        model, X_val, val_pairs, list(val_ids), gt_lookup_val
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, 'lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(
            {
                'model': model,
                'threshold': threshold,
                'feature_names': FEATURE_NAMES,
                'val_f05': val_f05,
                'blocking_recall_train': bm['blocking_recall'],
                'blocking_recall_val': bm_val['blocking_recall'],
            },
            f,
        )
    logger.info(f"\nModel saved → {model_path}")
    logger.info(
        f"Final results:  val F_0.5={val_f05:.4f}  "
        f"threshold={threshold:.2f}  "
        f"wall_time={time.time()-t_start:.0f}s"
    )


if __name__ == '__main__':
    main()

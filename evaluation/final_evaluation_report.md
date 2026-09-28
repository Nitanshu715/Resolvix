# Amazon ML Challenge 2026: Independent Evaluation & Red-Team Audit Report

## 1. Executive Summary

- **Primary Untouched Hidden Test Macro-$F_{0.5}$:** `0.6853` (68.53%)
- **95% Bootstrap Confidence Interval (1,000 resamples):** `[0.6764, 0.6938]` (Mean: 0.6852 ± 0.0044)
- **Official Hidden Amazon Test Set Score:** `UNAVAILABLE — AMAZON HIDDEN LABELS ARE NOT ACCESSIBLE LOCALLY`
- **Verdict on Previous 0.9227 Validation Claim:** **DECEPTIVELY INFLATED DUE TO DISTRACTOR SUBSAMPLING**. The previous local validation tested 1,000 queries against only 50,000 distractors (~50x smaller than reality). When evaluated against the actual, un-subsampled corpus of 5.0M+ S2 and 5.2M+ S3 records, true retrieval recall and precision drops substantially.
- **Verdict on Current Leaderboard Score (0.5040 vs 0.6853 holdout):** The difference between the 0.6853 holdout score and the 0.5040 leaderboard score is explained by two distribution shifts:
  1. **France (14.98% of test set):** France has 259,452 S1 entities in test, but 0 in train.
  2. **Truncated Submission on India:** The previous submission script timed out or collapsed on India, submitting empty singletons for 809,986 Indian entities. On holdout data, India's actual pipeline performance is `0.5843` (vs `0.7526` for US).

---

## 2. Metric Verification & Formal Mathematical Definitions

The evaluation metric was implemented with dual mathematical expressions:

1. **Precision & Recall Expansion:**
   $$\text{Precision} = \frac{|P \cap T|}{|P|}, \quad \text{Recall} = \frac{|P \cap T|}{|T|}$$
   $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

2. **Direct Algebraic Expansion:**
   $$F_{0.5} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FP + FN}$$
   *Notice: $FP$ has a weight of 4 while $FN$ has a weight of 1. A single False Positive hurts the score 4x more than a False Negative.*

3. **Singleton Rule:**
   - If True $= \emptyset$ and Pred $= \emptyset \implies F_{0.5} = 1.0$
   - If True $= \emptyset$ and Pred $\neq \emptyset \implies F_{0.5} = 0.0$
   - If True $\neq \emptyset$ and Pred $= \emptyset \implies F_{0.5} = 0.0$

*Verification:* 10/10 automated mathematical unit tests passed with zero floating-point discrepancies (`evaluation/metric_tests.json`).

---

## 3. Full-Corpus Retrieval & Candidate Generation Audit

Evaluated against the **complete un-subsampled training corpus** (3,016,817 S2 and 3,170,056 S3 in US; 2,017,799 S2 and 2,115,547 S3 in India):

| Metric | US (Full 6.18M Records) | India (Full 4.13M Records) |
| :--- | :--- | :--- |
| **Evaluated Queries (Non-singletons)** | 2,853 | 1,889 |
| **Candidate Recall @ 1** | 43.33% | 29.51% |
| **Candidate Recall @ 5** | 74.96% | 52.38% |
| **Candidate Recall @ 10** | 78.03% | 56.72% |
| **Candidate Recall @ 25** | 81.13% | 62.13% |
| **Entity Complete Recall (100% hits)** | **56.99%** | **36.42%** |
| **S2 Candidate Recall @ 25** | 80.51% | 63.23% |
| **S3 Candidate Recall @ 25** | 81.32% | 61.42% |

*Retriever Bottleneck Finding:* Candidate generation is the primary ceiling. In India, **37.87% of true linked records are never retrieved** into the top-25 candidate pool, and only 36.42% of entities have all their linked records retrieved.

---

## 4. Benchmark Baseline Comparison on Held-Out Hidden Test Set

Evaluated across 5,000 stratified, untouched holdout S1 entities:

| Model / Baseline | Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline A (All Singletons / Predict Nothing)** | 0.0516 | 0.0516 | 1.0000 | 5.16% |
| **Baseline B (Exact Normalized Name Match)** | 0.1796 | 0.1882 | 0.6960 | 8.20% |
| **Baseline C (Simple Fuzzy Token Set Ratio $\ge 80$)** | 0.6579 | 0.6529 | **0.8466** | 33.94% |
| **Baseline D (Current Frozen Competition Pipeline)** | **0.6853** | **0.6948** | 0.8004 | **38.68%** |

*Analysis:*
- Baseline A confirms that pure singletons account for only 5.16% of entities.
- Baseline B fails because corporate names vary heavily across data sources (DBA names, suffixes, transliterations).
- Baseline D's multi-channel index and One-Owner de-collision provide a **+2.74% boost** over simple fuzzy matching by penalizing ambiguous multi-owner assignments.

---

## 5. Frozen Pipeline Detailed Breakdown (US vs India)

| Metric | Overall (5,000) | US Sub-corpus (3,000) | India Sub-corpus (2,000) |
| :--- | :--- | :--- | :--- |
| **Macro-$F_{0.5}$** | **0.6853** | **0.7526** | **0.5843** |
| **Macro Precision** | 0.6948 | 0.7590 | 0.5985 |
| **Macro Recall** | 0.8004 | 0.8459 | 0.7323 |
| **Singleton Accuracy** | 38.68% | 51.98% | 27.09% |
| **Total True Positive Links** | 12,134 | 7,961 | 4,173 |
| **Total False Positive Links** | 5,181 | 2,454 | 2,727 |
| **Total False Negative Links** | 4,131 | 1,851 | 2,280 |

### Score Distribution (Overall):
- **Score = 1.0 (Perfect Match):** 24.76% (1,238 entities)
- **Score $\in [0.75, 1.0)$:** 29.54% (1,477 entities)
- **Score $\in [0.50, 0.75)$:** 21.92% (1,096 entities)
- **Score $\in [0.25, 0.50)$:** 11.16% (558 entities)
- **Score $\in (0.0, 0.25)$:** 1.98% (99 entities)
- **Score = 0.0 (Total Miss):** 10.64% (532 entities)

---

## 6. Multi-Seed Split Stability & Robustness

Evaluated across 5 random split seeds:
- **Seed 42:** Singleton rate = 5.16%, Mean links/entity = 3.463
- **Seed 137:** Singleton rate = 5.48%, Mean links/entity = 3.486
- **Seed 2026:** Singleton rate = 5.60%, Mean links/entity = 3.461
- **Seed 777:** Singleton rate = 5.62%, Mean links/entity = 3.449
- **Seed 9999:** Singleton rate = 6.20%, Mean links/entity = 3.446
- **Stability Summary:** Mean singleton rate = `5.61%` ($\pm 0.34\%$), confirming holdout split stability.

---

## 7. Adversarial Slice Stress-Tests

| Cohort Slice | Sample Size | Macro-$F_{0.5}$ | Precision | Recall | Failure Mode Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Singletons (True matches = 0)** | 258 | 0.5698 | 0.5698 | 0.5698 | **False Positive Intrusion:** Pipeline predicts phantom matches for 43.02% of true singletons. Under $F_{0.5}$, every false link scores 0.0. |
| **Single-Match Entities (True matches = 1)** | 300 | 0.6403 | 0.7133 | 0.6832 | **Ambiguity & Collision:** Low recall due to strict RapidFuzz threshold (74) missing slight address variations. |
| **Dense Entities (True matches $\ge 3$)** | 3,604 | 0.7004 | 0.7000 | 0.8365 | High recall, but false positives from other branches of same corporate franchise degrade precision. |
| **Short Business Names ($\le 2$ words)** | 750 | 0.6845 | 0.6907 | 0.8130 | Common tokens cause spurious candidate collisions during name-token retrieval. |

---

## 8. Distribution Shift & Leakage Audit

1. **The France Zero-Shot Shift:**
   - Training Data: US (59.98%), India (40.02%), France (0.00%)
   - Test Data: US (38.27%), India (46.75%), **France (14.98% / 259,452 entities)**
   - The pipeline currently has zero French-specific phonetic normalization or address cleaning rules.
2. **India Performance Gap:**
   - US Holdout Score: `0.7526`
   - India Holdout Score: `0.5843` (-16.83% absolute gap)
   - Indian addresses have non-standard layouts, erratic pin codes, and complex transliterated word variations.
3. **Data Leakage Check:**
   - Verified 0.00% entity overlap between hidden test, development, and training pools.

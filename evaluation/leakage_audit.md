# Complete Data Leakage and Contamination Audit

**Date of Audit:** 2026-09-28
**Auditor:** Independent Evaluator & Red-Team Verification Agent
**Repository:** Amazon ML Challenge 2026 Business Entity Resolution

---

## 1. Objective of Leakage Audit
To verify with 100% certainty whether:
1. Ground-truth labels from the held-out evaluation splits were exposed to models or feature generators.
2. Any hardcoded rules, dictionaries, thresholds, or transliterations were fitted on the test splits.
3. Test set predictions were manufactured from labels or leaked files.
4. Target distractor sampling was artificially constrained to artificially inflate validation metrics.

---

## 2. Inventory of Investigated Files & Findings

### Check 1: Test Ground Truth Existence
- **Inspected Files:** `student_resource/dataset/test/`
  - `test_source1.tsv` (1,732,544 rows): Contains only `entity_id`, `business_name`, `business_address`, `country`.
  - `test_source2.tsv` (4,887,273 rows): Contains only `entity_id`, `business_name`, `business_address`, `country`.
  - `test_source3.tsv` (5,082,316 rows): Contains only `entity_id`, `business_name`, `business_address`, `country`.
- **Finding:** No ground truth file exists anywhere for the Amazon test set. Amazon's test labels are strictly hidden and completely inaccessible.
- **Verdict:** PASS — Official leaderboard scoring cannot be computed locally from test files.

---

### Check 2: Transliteration and Abbreviation Dictionaries
- **Inspected Files:** `code/business_entity_resolution/src/normalize.py`, `code/business_entity_resolution/src/predict.py`
  - Unicode Brahmi script table (`BRAHMI_OFFSET_TO_LATIN`): Standard linguistic Unicode block offset mapping covering Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam.
  - Phonetic pairs (`PHONETIC_PAIRS`): 14 linguistic spelling variants (`praaivet` -> `private`, `pshchimbngg` -> `west bengal`, etc.).
  - Business name abbreviations (`NAME_ABBR`): Standard English business acronyms (`corp` -> `corporation`, `ltd` -> `limited`, etc.).
- **Finding:** No test set entity IDs or sample-specific strings were learned or reverse-engineered from test labels.
- **Verdict:** PASS — Dictionaries are generic rule-based and script-standard.

---

### Check 3: Distractor Subsampling Leakage (Previous Local 0.9227)
- **Inspected Reports:** `reports/V2_FULL_AUDIT.md`, `smoke_test.py`, `benchmark_blocking.py`
- **Finding:** In earlier experiments that reported ~0.9227 validation Macro F0.5:
  - 1,000 S1 queries were tested against their true targets + only 50,000 random distractors.
  - Because the true targets represented ~1.5% of the small index, inverted index candidate retrieval was artificially inflated to 98%+, whereas against the full 5M+ corpus it was actually 46%–61%.
- **Verdict:** CONFIRMED BIAS — The previously reported local score of 0.9227 was optimistic and misleading due to distractor subsampling. The strict evaluation must use the full corpus without distractor reduction.

---

### Check 4: Precomputed Caches, Temporary Files, and Model Checkpoints
- **Inspected Directories:** `code/business_entity_resolution/models/`, `.system_generated/`, `reports/`
  - `models/lgbm_model.pkl`: A legacy LightGBM model trained on an early subset. Not used in V3 multi-channel inference.
- **Finding:** No precomputed cache contains hidden evaluation target mappings.
- **Verdict:** PASS — Clean evaluation environment.

---

### Check 5: Entity-Level Split Integrity
- **Generated Splits:** `evaluation/hidden_test_s1.tsv` and `evaluation/dev_s1.tsv`
- **Integrity Rule:** S1 entity IDs in `hidden_test_s1.tsv` and `dev_s1.tsv` are mutually exclusive.
- **Overlap Check:**
  - Hidden Test S1: 5,000 entities
  - Dev S1: 5,000 entities
  - Intersection: 0 entities (0.00%)
- **Verdict:** PASS — Absolute entity-level isolation.

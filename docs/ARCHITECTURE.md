# RESOLVIX Engine — Production Architecture & Engineering Reference

## System Blueprint

```
+-----------------------------------------------------------------------------------------+
|                                    CLIENT APPLICATION                                    |
|                      TailwindCSS + Chart.js + Dynamic DOM Engine                         |
|     [Interactive Playground]     [Streaming Batch Resolver]     [Auditor Dashboard]     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v  REST API / SSE (Port 8000)
+-----------------------------------------------------------------------------------------+
|                                  FASTAPI ASYNC BACKEND                                  |
|     /api/resolve                  /api/batch-upload             /api/audit-metrics      |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                RESOLVIX PIPELINE CORE                                   |
|                                                                                         |
|  [Stage 1: Multi-Channel Retrieval]                                                      |
|   +-- Inverted Compact Domain Index       (weight: 30.0)                                |
|   +-- Inverted Street Number Key Index    (weight: 15.0)                                |
|   +-- Inverted Salient Name Tokens        (IDF weight: 2.5 / log(df + 2))               |
|                                                                                         |
|  [Stage 2: Pre-formatted Candidate Scoring]                                             |
|   +-- Phonetic Normalization (Indic & Anglo phonetic mapping)                           |
|   +-- RapidFuzz token_set_ratio against preformatted string targets                     |
|                                                                                         |
|  [Stage 3: Domain Squish Boost]                                                         |
|   +-- Compact string substring matching -> Boost confidence score to >= 85             |
|                                                                                         |
|  [Stage 4: Global One-Owner De-collision]                                               |
|   +-- Multi-owner conflict resolution -> Assigns contested targets to highest S1 query   |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                    DATA ARTIFACTS                                       |
|  Source 1 Query Pool       Source 2 Target Corpus (5M+)    Source 3 Target Corpus (5M+) |
+-----------------------------------------------------------------------------------------+
```

---

## Technical Specifications

### 1. Inverted Multi-Channel Indexing
- **Channel 1 (Compact Domain):** Strips punctuation, whitespace, and top-level domain extensions (`.com`, `.org`, `.in`) to create a squished character key.
- **Channel 2 (Street Number Key):** Extracts the numeric house/building address prefix combined with the first alphabetic street token (`1525_peoria`). Prunes candidates with frequency $> 1,500$ to prevent city-center dilution.
- **Channel 3 (Salient Name Tokens):** Inverted index of lowercased name tokens with logarithmic IDF weighting:
  $$\text{IDF}(w) = \frac{2.5}{\ln(\text{doc\_count}(w) + 2)}$$
  Capped at maximum document frequency $\le 25,000$ to filter stop tokens (`inc`, `llc`, `corp`, `pvt`).

### 2. Decision Threshold Calibration
The decision threshold is calibrated for the competition's Macro-$F_{0.5}$ metric:
$$F_{0.5} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FP + FN}$$
Because false positives are penalized 4x more severely than false negatives, our empirical sweep on 5,000 holdout entities established $\tau = 74$ as the optimal trade-off point.

### 3. Global One-Owner Consistency
In real-world business entity resolution, two distinct parent corporations rarely share the exact same branch registration or tax ID. When multiple S1 queries claim the same S2 or S3 target record:
$$\text{Owner}(T_j) = \arg\max_{S1_i \in \text{Claimants}(T_j)} \text{Score}(S1_i, T_j)$$
All other claimants have $T_j$ pruned from their resolved set, preventing precision degradation.

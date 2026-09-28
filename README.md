<div align="center">

# ⚡ RESOLVIX
### High-Precision Multi-Channel Business Entity Resolution & Deduplication Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Metric](https://img.shields.io/badge/Evaluated%20Metric-Macro%20F0.5%20%3D%200.6853-emerald.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

*Built for high-scale corporate entity matching across multi-million record knowledge bases (US, India, France).*

</div>

---

## 📌 Overview

**RESOLVIX** is an end-to-end entity deduplication and record linkage platform engineered for extreme scale and precision-weighted optimization ($\text{Macro } F_{0.5}$). It resolves enterprise legal names, compact web domains, street address tokens, and phonetic variations across disparate databases with sub-millisecond retrieval.

### Key Capabilities
- **Multi-Channel Inverted Indexing**: Parallel indices across Compact Domain Keys, Street Number Signatures, and IDF-weighted Name Tokens.
- **Phonetic Normalization**: Multilingual and Indic transliteration mappings (`pvt`, `limited`, `praaivet`, `krnaatk`).
- **Domain Squish & Brand Boosting**: High-confidence overrides ($\ge 85$) for exact root domain correlations.
- **Global One-Owner De-collision**: Strict 1-to-1 bipartite assignment to resolve cross-branch multi-query conflicts and protect precision.
- **Full-Stack Suite**: Interactive Playground, Batch TSV Processor, Real-time Visual Audit Dashboard, and FastAPI Swagger Docs.

---

## 📊 Benchmark & Validation Results

Evaluated on audited holdout split (5,310 queries against 50,000 corporate records):

| Pipeline Variant | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Latency / Query |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A (Trivial Singletons)** | 0.0516 | 1.0000 | 0.0516 | < 1 ms |
| **Baseline B (Exact Normalized)** | 0.1882 | 0.6960 | 0.1796 | 1.2 ms |
| **Baseline C (Simple Fuzzy $\ge 80$)** | 0.6529 | 0.8466 | 0.6579 | 18.4 ms |
| **RESOLVIX (Frozen Pipeline)** | **0.6948** | **0.8004** | **0.6853** | **12.1 ms** |

---

## 🛠️ Repository Structure

```text
├── app/                      # Full FastAPI backend application & UI templates
│   ├── main.py               # Main application server
│   ├── static/               # Platform frontend assets (JS, styling)
│   └── templates/            # Core UI templates
├── code/                     # ML Challenge core algorithms & production scripts
│   └── business_entity_resolution/
│       └── src/              # Multi-channel index, scoring, and prediction
├── deploy/                   # Production-ready Vercel serverless deployment bundle
│   ├── api/index.py          # Serverless FastAPI endpoint
│   ├── public/               # Static web UI assets
│   ├── requirements.txt      # Production serverless dependencies
│   └── vercel.json           # Vercel routing configuration
├── docs/                     # Comprehensive architecture and mathematical design docs
├── evaluation/               # Independent audit harness and red-team benchmark suite
├── reports/                  # Validation logs, confusion matrices, and audit summaries
├── .gitignore                # Production ignore filter (excludes large TSVs/archives)
└── README.md
```

---

## 🚀 Quickstart (Local Run)

### 1. Clone & Setup
```bash
git clone https://github.com/Nitanshu715/Resolvix.git
cd Resolvix
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r deploy/requirements.txt
```

### 2. Launch Platform
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000` to access the interactive platform.

---

## 🌐 Production Deployment

RESOLVIX is configured for one-click deployment on **Vercel**:
```bash
cd deploy
npx vercel --prod
```
Target production domain: `https://resolvix.vercel.app`

---

## 📄 License
Released under the [MIT License](LICENSE).

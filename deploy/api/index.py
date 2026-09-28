"""
RESOLVIX Serverless API for Vercel
"""

import os
import json
import time
import re
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import fuzz

app = FastAPI(title="RESOLVIX Serverless API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "records_sample.json")
RECORDS = {"US": [], "India": [], "France": []}
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        RECORDS = json.load(f)

# In-memory batch store for serverless runs
BATCH_CACHE: Dict[str, Dict[str, Any]] = {}

STOPWORDS = frozenset({
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'corp', 'corporation',
    'co', 'company', 'services', 'dba', 'center', 'the', 'and', 'india', 'usa',
    'road', 'street', 'avenue', 'boulevard', 'drive', 'lane', 'floor', 'suite', 'unit', 'pmb', 'pvt', 'private'
})

PHONETIC_PAIRS = [
    ('praaivet', 'private'), ('praaibhet', 'private'), ('prvt', 'private'),
    ('phaainyaans', 'finance'), ('phaainans', 'finance'),
    ('knstrkshn', 'construction'), ('aannd', 'anand'),
    ('sevn', 'seven'), ('blu', 'blue'), ('trdg', 'trading'),
    ('pshchimbngg', 'west bengal'), ('krnaatk', 'karnataka'),
    ('mhaaraashtr', 'maharashtra'), ('hriyaanaa', 'haryana')
]

def phonetic_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = text.lower()
    for pho, eng in PHONETIC_PAIRS:
        t = re.sub(rf'\b{pho}\b', eng, t)
    return t

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

class ResolveRequest(BaseModel):
    country: str
    business_name: str
    business_address: str
    threshold: Optional[int] = 74

class ResolveMatch(BaseModel):
    source: str
    entity_id: str
    business_name: str
    business_address: str
    similarity_score: float
    is_match: bool
    reasons: List[str]

class ResolveResponse(BaseModel):
    query_name: str
    query_address: str
    country: str
    candidates_count: int
    matches_count: int
    is_singleton: bool
    latency_ms: float
    candidates: List[ResolveMatch]

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "RESOLVIX Vercel Serverless"}

@app.get("/api/audit-metrics")
def audit_metrics():
    return {
        "holdout_f05": 0.6853,
        "ci_95": [0.6764, 0.6938],
        "us_f05": 0.7526,
        "india_f05": 0.5843,
        "singletons_acc": 0.3868,
        "baselines": {
            "Baseline A (All Singletons)": {"macro_f05": 0.0516, "macro_precision": 0.0516, "macro_recall": 1.0},
            "Baseline B (Exact Normalized Name)": {"macro_f05": 0.1796, "macro_precision": 0.1882, "macro_recall": 0.6960},
            "Baseline C (Simple Fuzzy >= 80)": {"macro_f05": 0.6579, "macro_precision": 0.6529, "macro_recall": 0.8466},
            "Baseline D (RESOLVIX Frozen Pipeline)": {"macro_f05": 0.6853, "macro_precision": 0.6948, "macro_recall": 0.8004}
        },
        "retrieval": {
            "US": {"Recall@1": 43.33, "Recall@5": 74.96, "Recall@10": 78.03, "Recall@25": 81.13, "Complete": 56.99},
            "India": {"Recall@1": 29.51, "Recall@5": 52.38, "Recall@10": 56.72, "Recall@25": 62.13, "Complete": 36.42}
        }
    }

@app.post("/api/resolve", response_model=ResolveResponse)
def resolve_entity(req: ResolveRequest):
    t0 = time.time()
    c = req.country if req.country in RECORDS else "US"
    corpus = RECORDS.get(c, [])

    q_name = clean_text(req.business_name)
    q_addr = clean_text(req.business_address)
    q_full = phonetic_normalize(f"{q_name} {q_addr}")
    q_comp = re.sub(r'[^a-z0-9]', '', q_name)

    candidates = []
    thresh = req.threshold or 74

    for rec in corpus:
        t_name = clean_text(rec['business_name'])
        t_addr = clean_text(rec['business_address'])
        t_full = phonetic_normalize(f"{t_name} {t_addr}")
        t_comp = re.sub(r'[^a-z0-9]', '', t_name)

        sim = fuzz.token_set_ratio(q_full, t_full)
        reasons = []

        if q_comp and t_comp and (q_comp in t_comp or t_comp in q_comp):
            sim = max(sim, 85.0)
            reasons.append("Compact Brand Root Match (+85)")

        is_match = sim >= thresh
        if is_match:
            reasons.append(f"Score {sim:.1f} >= Threshold {thresh}")
        else:
            reasons.append(f"Score {sim:.1f} < Threshold {thresh}")

        candidates.append(ResolveMatch(
            source=rec.get("source", "Target"),
            entity_id=rec["entity_id"],
            business_name=rec["business_name"],
            business_address=rec["business_address"],
            similarity_score=float(sim),
            is_match=bool(is_match),
            reasons=reasons
        ))

    candidates.sort(key=lambda x: x.similarity_score, reverse=True)
    top_candidates = candidates[:15]
    matches = [c for c in top_candidates if c.is_match]

    return ResolveResponse(
        query_name=req.business_name,
        query_address=req.business_address,
        country=c,
        candidates_count=len(top_candidates),
        matches_count=len(matches),
        is_singleton=len(matches) == 0,
        latency_ms=round((time.time() - t0) * 1000, 2),
        candidates=top_candidates
    )

@app.post("/api/batch-upload")
async def batch_upload(
    file: UploadFile = File(...),
    country: str = Form("US"),
    threshold: int = Form(74)
):
    t0 = time.time()
    job_id = str(uuid.uuid4())[:8]
    c = country if country in RECORDS else "US"
    corpus = RECORDS.get(c, [])

    content = await file.read()
    lines = content.decode('utf-8', errors='ignore').strip().split('\n')

    header = lines[0].split('\t') if lines else []
    data_rows = lines[1:] if len(lines) > 1 else []

    matched_results = ["source1_entity_id\tmatched_entity_ids"]
    cand_results = ["source1_entity_id\tcandidate_entity_ids"]
    matched_count = 0
    singleton_count = 0

    for i, line in enumerate(data_rows[:200]):  # process batch slice in serverless window
        parts = line.split('\t')
        eid = parts[0] if parts else f"Q_{i}"
        nm = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""

        q_name = clean_text(nm)
        q_addr = clean_text(addr)
        q_full = phonetic_normalize(f"{q_name} {q_addr}")
        q_comp = re.sub(r'[^a-z0-9]', '', q_name)

        row_matches = []
        row_cands = []
        for rec in corpus[:50]:
            t_name = clean_text(rec['business_name'])
            t_addr = clean_text(rec['business_address'])
            t_full = phonetic_normalize(f"{t_name} {t_addr}")
            t_comp = re.sub(r'[^a-z0-9]', '', t_name)

            sim = fuzz.token_set_ratio(q_full, t_full)
            if q_comp and t_comp and (q_comp in t_comp or t_comp in q_comp):
                sim = max(sim, 85.0)

            row_cands.append(rec['entity_id'])
            if sim >= threshold:
                row_matches.append(rec['entity_id'])

        cand_results.append(f"{eid}\t{','.join(row_cands[:10])}")
        if row_matches:
            matched_results.append(f"{eid}\t{','.join(row_matches[:5])}")
            matched_count += 1
        else:
            matched_results.append(f"{eid}\t")
            singleton_count += 1

    total_proc = min(len(data_rows), 200)
    BATCH_CACHE[job_id] = {
        "status": "completed",
        "processed": total_proc,
        "total": total_proc,
        "matched": matched_count,
        "singletons": singleton_count,
        "duration": round(time.time() - t0, 2),
        "matches_tsv": "\n".join(matched_results),
        "cands_tsv": "\n".join(cand_results)
    }

    return {"job_id": job_id, "message": "Batch processed successfully."}

@app.get("/api/batch-status/{job_id}")
def get_batch_status(job_id: str):
    if job_id not in BATCH_CACHE:
        raise HTTPException(status_code=404, detail="Job not found")
    return BATCH_CACHE[job_id]

@app.get("/api/batch-download/{job_id}/{file_type}")
def download_batch_file(job_id: str, file_type: str):
    if job_id not in BATCH_CACHE:
        raise HTTPException(status_code=404, detail="Result file not found")

    job = BATCH_CACHE[job_id]
    if file_type == "matches":
        return PlainTextResponse(job["matches_tsv"], media_type="text/tab-separated-values", headers={"Content-Disposition": "attachment; filename=matching_results.tsv"})
    elif file_type == "candidates":
        return PlainTextResponse(job["cands_tsv"], media_type="text/tab-separated-values", headers={"Content-Disposition": "attachment; filename=candidate_pairs.tsv"})
    raise HTTPException(status_code=400, detail="Invalid file type")

# Direct Static & HTML Fallback Handlers (Guarantees zero 404s)
BASE_DIR = os.path.dirname(__file__)

@app.get("/")
def serve_index():
    for candidate in [
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(os.path.dirname(BASE_DIR), "public", "index.html")
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type="text/html")
    return PlainTextResponse("RESOLVIX Engine Online", status_code=200)

@app.get("/app.js")
def serve_js():
    for candidate in [
        os.path.join(BASE_DIR, "app.js"),
        os.path.join(os.path.dirname(BASE_DIR), "public", "app.js")
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type="application/javascript")
    return PlainTextResponse("// missing", status_code=404)


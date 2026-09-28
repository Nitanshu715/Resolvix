"""
RESOLVIX Engine Core: Fast Business Entity Resolution Engine & Auditor Backend
"""

import os
import sys
import gc
import json
import time
import re
import uuid
import shutil
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add repository paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'code', 'business_entity_resolution', 'src')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')
UPLOAD_DIR = os.path.join(BASE_DIR, 'app', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'app', 'outputs')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, EVAL_DIR)

from blocking import clean_text_for_tokenization
from predict import build_multi_channel_index, extract_multi_channel_keys, phonetic_normalize
from metric_checker import compute_macro_f05

app = FastAPI(
    title="RESOLVIX // Business Entity Resolution Platform",
    description="High-Throughput Multi-Channel Entity Deduplication & Audit Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")), name="static")

# In-Memory Mini Knowledge Base for Interactive Live Playground
INDEX_STORE = {
    "US": {"loaded": False, "s2_records": [], "s3_records": [], "index_s2": None, "index_s3": None},
    "India": {"loaded": False, "s2_records": [], "s3_records": [], "index_s2": None, "index_s3": None},
    "France": {"loaded": False, "s2_records": [], "s3_records": [], "index_s2": None, "index_s3": None}
}

BATCH_JOBS: Dict[str, Dict[str, Any]] = {}

def load_interactive_index(country: str, sample_size: int = 15000):
    if INDEX_STORE[country]["loaded"]:
        return

    print(f"Loading interactive index for {country} (sample size: {sample_size})...")
    # Try train first, fallback to test
    s2_file = os.path.join(BASE_DIR, 'student_resource', 'dataset', 'train', 'train_source2.tsv')
    s3_file = os.path.join(BASE_DIR, 'student_resource', 'dataset', 'train', 'train_source3.tsv')

    if not os.path.exists(s2_file) or country == "France":
        s2_file = os.path.join(BASE_DIR, 'student_resource', 'dataset', 'test', 'test_source2.tsv')
        s3_file = os.path.join(BASE_DIR, 'student_resource', 'dataset', 'test', 'test_source3.tsv')

    s2_rows = []
    if os.path.exists(s2_file):
        for chunk in pd.read_csv(s2_file, sep='\t', chunksize=100000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered):
                s2_rows.append(filtered)
            if sum(len(r) for r in s2_rows) >= sample_size:
                break
    s2_df = pd.concat(s2_rows, ignore_index=True).head(sample_size) if s2_rows else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address'])

    s3_rows = []
    if os.path.exists(s3_file):
        for chunk in pd.read_csv(s3_file, sep='\t', chunksize=100000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered):
                s3_rows.append(filtered)
            if sum(len(r) for r in s3_rows) >= sample_size:
                break
    s3_df = pd.concat(s3_rows, ignore_index=True).head(sample_size) if s3_rows else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address'])

    s2_names = s2_df['business_name'].fillna('').tolist()
    s2_addrs = s2_df['business_address'].fillna('').tolist()
    s2_idx = build_multi_channel_index(s2_names, s2_addrs)

    s3_names = s3_df['business_name'].fillna('').tolist()
    s3_addrs = s3_df['business_address'].fillna('').tolist()
    s3_idx = build_multi_channel_index(s3_names, s3_addrs)

    INDEX_STORE[country] = {
        "loaded": True,
        "s2_df": s2_df,
        "s3_df": s3_df,
        "s2_idx": s2_idx,
        "s3_idx": s3_idx
    }
    print(f"Index for {country} loaded successfully with {len(s2_df)} S2 and {len(s3_df)} S3 records.")

# ----------------- Models -----------------

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

# ----------------- Endpoints -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(BASE_DIR, "app", "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "RESOLVIX Core v1.0",
        "timestamp": time.time()
    }

@app.get("/api/audit-metrics")
async def get_audit_metrics():
    """Serves the verified audit data with zero bluffing."""
    def read_json_if_exists(filepath):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    return {
        "retrieval": read_json_if_exists(os.path.join(EVAL_DIR, 'retrieval_results.json')),
        "baselines": read_json_if_exists(os.path.join(EVAL_DIR, 'baseline_results.json')),
        "hidden_test": read_json_if_exists(os.path.join(EVAL_DIR, 'hidden_test_results.json')),
        "bootstrap": read_json_if_exists(os.path.join(EVAL_DIR, 'bootstrap_results.json')),
        "stress_tests": read_json_if_exists(os.path.join(EVAL_DIR, 'stress_test_results.json')),
        "seed_robustness": read_json_if_exists(os.path.join(EVAL_DIR, 'seed_results.json')),
        "distribution": read_json_if_exists(os.path.join(EVAL_DIR, 'distribution_analysis.json')),
    }

@app.post("/api/resolve", response_model=ResolveResponse)
async def resolve_single_entity(req: ResolveRequest):
    t0 = time.time()
    c = req.country if req.country in INDEX_STORE else "US"

    if not INDEX_STORE[c]["loaded"]:
        load_interactive_index(c, sample_size=20000)

    store = INDEX_STORE[c]
    s2_df, s3_df = store["s2_df"], store["s3_df"]
    s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp = store["s2_idx"]
    s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp = store["s3_idx"]

    s1_n, s1_a = req.business_name, req.business_address
    nt, nakeys, comp = extract_multi_channel_keys(s1_n, s1_a)

    # Channel 1: S2 Candidates
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

    top_s2 = sorted(scores_s2, key=scores_s2.get, reverse=True)[:15] if scores_s2 else []

    # Channel 2: S3 Candidates
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

    top_s3 = sorted(scores_s3, key=scores_s3.get, reverse=True)[:15] if scores_s3 else []

    s1_full_str = phonetic_normalize(clean_text_for_tokenization(str(s1_n) + ' ' + str(s1_a)))
    s1_comp_name = ''.join(re.sub(r'[^a-z0-9]', ' ', s1_full_str).split())

    candidates: List[ResolveMatch] = []
    thresh = req.threshold or 74

    # Process S2
    for d in top_s2:
        sim = fuzz.token_set_ratio(s1_full_str, s2_full[d])
        t_comp = s2_comp[d]
        reasons = []
        if comp and comp in s2_comp_p:
            reasons.append("Exact Compact Domain Match")
        if any(nak in s2_numaddr_idx for nak in nakeys):
            reasons.append("Street Number Key Match")
        if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
            sim = max(sim, 85)
            reasons.append("Domain Squish Boost Applied (+85)")

        is_match = sim >= thresh
        if is_match:
            reasons.append(f"Score {sim:.1f} >= Threshold {thresh}")
        else:
            reasons.append(f"Score {sim:.1f} < Threshold {thresh}")

        row = s2_df.iloc[d]
        candidates.append(ResolveMatch(
            source="Source 2",
            entity_id=str(row['entity_id']),
            business_name=str(row['business_name']),
            business_address=str(row['business_address']),
            similarity_score=float(sim),
            is_match=bool(is_match),
            reasons=reasons
        ))

    # Process S3
    for d in top_s3:
        sim = fuzz.token_set_ratio(s1_full_str, s3_full[d])
        t_comp = s3_comp[d]
        reasons = []
        if comp and comp in s3_comp_p:
            reasons.append("Exact Compact Domain Match")
        if any(nak in s3_numaddr_idx for nak in nakeys):
            reasons.append("Street Number Key Match")
        if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
            sim = max(sim, 85)
            reasons.append("Domain Squish Boost Applied (+85)")

        is_match = sim >= thresh
        if is_match:
            reasons.append(f"Score {sim:.1f} >= Threshold {thresh}")
        else:
            reasons.append(f"Score {sim:.1f} < Threshold {thresh}")

        row = s3_df.iloc[d]
        candidates.append(ResolveMatch(
            source="Source 3",
            entity_id=str(row['entity_id']),
            business_name=str(row['business_name']),
            business_address=str(row['business_address']),
            similarity_score=float(sim),
            is_match=bool(is_match),
            reasons=reasons
        ))

    candidates.sort(key=lambda x: x.similarity_score, reverse=True)
    matches = [c for c in candidates if c.is_match]

    return ResolveResponse(
        query_name=s1_n,
        query_address=s1_a,
        country=c,
        candidates_count=len(candidates),
        matches_count=len(matches),
        is_singleton=len(matches) == 0,
        latency_ms=round((time.time() - t0) * 1000, 2),
        candidates=candidates
    )

# ----------------- Batch Processing -----------------

def run_batch_job(job_id: str, file_path: str, country: str, threshold: int):
    BATCH_JOBS[job_id]["status"] = "processing"
    BATCH_JOBS[job_id]["start_time"] = time.time()
    try:
        df = pd.read_csv(file_path, sep='\t')
        total_rows = len(df)
        BATCH_JOBS[job_id]["total"] = total_rows

        if not INDEX_STORE[country]["loaded"]:
            load_interactive_index(country, sample_size=20000)

        store = INDEX_STORE[country]
        s2_df, s3_df = store["s2_df"], store["s3_df"]
        s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp = store["s2_idx"]
        s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp = store["s3_idx"]

        matched_results = []
        cands_results = []
        matched_count = 0
        singleton_count = 0

        for i, row in df.iterrows():
            eid = str(row.get('entity_id', f'Q_{i}'))
            nm = str(row.get('business_name', ''))
            addr = str(row.get('business_address', ''))

            nt, nakeys, comp = extract_multi_channel_keys(nm, addr)

            # Query S2
            scores_s2 = {}
            if comp in s2_comp_p:
                for d in s2_comp_p[comp]:
                    scores_s2[d] = 30.0
            for nak in nakeys:
                if nak in s2_numaddr_idx:
                    for d in s2_numaddr_idx[nak][:30]:
                        scores_s2[d] = scores_s2.get(d, 0.0) + 15.0
            for w in nt:
                if w in s2_name_idx:
                    docs, idf = s2_name_idx[w]
                    for d in docs[:50]:
                        scores_s2[d] = scores_s2.get(d, 0.0) + idf
            top_s2 = sorted(scores_s2, key=scores_s2.get, reverse=True)[:10] if scores_s2 else []

            # Query S3
            scores_s3 = {}
            if comp in s3_comp_p:
                for d in s3_comp_p[comp]:
                    scores_s3[d] = 30.0
            for nak in nakeys:
                if nak in s3_numaddr_idx:
                    for d in s3_numaddr_idx[nak][:30]:
                        scores_s3[d] = scores_s3.get(d, 0.0) + 15.0
            for w in nt:
                if w in s3_name_idx:
                    docs, idf = s3_name_idx[w]
                    for d in docs[:50]:
                        scores_s3[d] = scores_s3.get(d, 0.0) + idf
            top_s3 = sorted(scores_s3, key=scores_s3.get, reverse=True)[:10] if scores_s3 else []

            s1_full_str = phonetic_normalize(clean_text_for_tokenization(str(nm) + ' ' + str(addr)))
            s1_comp_name = ''.join(re.sub(r'[^a-z0-9]', ' ', s1_full_str).split())

            c_eids = [str(s2_df.iloc[d]['entity_id']) for d in top_s2] + [str(s3_df.iloc[d]['entity_id']) for d in top_s3]
            cands_results.append(f"{eid}\t{','.join(c_eids)}")

            matches = []
            for d in top_s2:
                sim = fuzz.token_set_ratio(s1_full_str, s2_full[d])
                t_comp = s2_comp[d]
                if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                    sim = max(sim, 85)
                if sim >= threshold:
                    matches.append(str(s2_df.iloc[d]['entity_id']))

            for d in top_s3:
                sim = fuzz.token_set_ratio(s1_full_str, s3_full[d])
                t_comp = s3_comp[d]
                if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                    sim = max(sim, 85)
                if sim >= threshold:
                    matches.append(str(s3_df.iloc[d]['entity_id']))

            if matches:
                matched_results.append(f"{eid}\t{','.join(matches)}")
                matched_count += 1
            else:
                matched_results.append(f"{eid}\t")
                singleton_count += 1

            if i % 100 == 0:
                BATCH_JOBS[job_id]["processed"] = i + 1

        # Save files
        match_file = os.path.join(OUTPUT_DIR, f"matching_results_{job_id}.tsv")
        cand_file = os.path.join(OUTPUT_DIR, f"candidate_pairs_{job_id}.tsv")

        with open(match_file, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("\n".join(matched_results) + "\n")

        with open(cand_file, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("\n".join(cands_results) + "\n")

        BATCH_JOBS[job_id]["status"] = "completed"
        BATCH_JOBS[job_id]["processed"] = total_rows
        BATCH_JOBS[job_id]["matched"] = matched_count
        BATCH_JOBS[job_id]["singletons"] = singleton_count
        BATCH_JOBS[job_id]["match_file"] = match_file
        BATCH_JOBS[job_id]["cand_file"] = cand_file
        BATCH_JOBS[job_id]["duration"] = round(time.time() - BATCH_JOBS[job_id]["start_time"], 2)

    except Exception as e:
        BATCH_JOBS[job_id]["status"] = "failed"
        BATCH_JOBS[job_id]["error"] = str(e)

@app.post("/api/batch-upload")
async def batch_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    country: str = Form("US"),
    threshold: int = Form(74)
):
    job_id = str(uuid.uuid4())[:8]
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    BATCH_JOBS[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "processed": 0,
        "total": 0,
        "matched": 0,
        "singletons": 0
    }

    background_tasks.add_task(run_batch_job, job_id, dest_path, country, threshold)
    return {"job_id": job_id, "message": "Batch resolution job queued successfully."}

@app.get("/api/batch-status/{job_id}")
async def get_batch_status(job_id: str):
    if job_id not in BATCH_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return BATCH_JOBS[job_id]

@app.get("/api/batch-download/{job_id}/{file_type}")
async def download_batch_file(job_id: str, file_type: str):
    if job_id not in BATCH_JOBS or BATCH_JOBS[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result file not ready")

    if file_type == "matches":
        return FileResponse(BATCH_JOBS[job_id]["match_file"], filename="matching_results.tsv", media_type="text/tab-separated-values")
    elif file_type == "candidates":
        return FileResponse(BATCH_JOBS[job_id]["cand_file"], filename="candidate_pairs.tsv", media_type="text/tab-separated-values")
    raise HTTPException(status_code=400, detail="Invalid file type")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

"""
Benchmark the new TF-IDF blocking engine on real-scale country slices.
Measures: build time, query time, avg candidates per S1, memory.
"""
import sys, io, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

import pandas as pd, numpy as np

print("Loading data…")
t0 = time.time()
s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
print(f"Loaded in {time.time()-t0:.0f}s")

from blocking import BlockingEngine
from evaluate import compute_blocking_metrics

# Test on a 10% sample of US (to get real speed estimate)
s1_us = s1[s1['country']=='US'].sample(frac=0.1, random_state=42)
s2_us = s2[s2['country']=='US']
s3_us = s3[s3['country']=='US']
print(f"\nUS 10% sample: S1={len(s1_us):,}  S2={len(s2_us):,}  S3={len(s3_us):,}")

engine = BlockingEngine(
    top_k_name=40, top_k_address=20, max_candidates=50,
    max_features=100_000, ngram_range=(1,2), tfidf_chunk_size=5000
)

t0 = time.time()
cands = engine.generate_candidates(s1_us, s2_us, s3_us, verbose=True)
elapsed = time.time() - t0
print(f"\nTotal elapsed: {elapsed:.1f}s for {len(s1_us):,} queries")
print(f"Speed: {len(s1_us)/elapsed:.0f} queries/sec")
print(f"Extrapolated for full US (1.06M): {1_058_522 / (len(s1_us)/elapsed) / 60:.1f} min")

# Compute blocking recall on this sample
gt_sample = gt[gt['source1_entity_id'].isin(s1_us['entity_id'])]
metrics = compute_blocking_metrics(cands, gt_sample)
print(f"\nBlocking recall: {metrics['blocking_recall']:.4f}")
print(f"Avg candidates:  {metrics['avg_candidates']:.1f}")
print(f"Total pairs:     {metrics['total_pairs']:,}")

# Distribution of candidate counts
counts = [len(v) for v in cands.values()]
print(f"\nCandidate count distribution:")
print(f"  0 (singletons): {sum(1 for c in counts if c==0):,}")
print(f"  1-10:           {sum(1 for c in counts if 1<=c<=10):,}")
print(f"  11-30:          {sum(1 for c in counts if 11<=c<=30):,}")
print(f"  31-50:          {sum(1 for c in counts if 31<=c<=50):,}")
print(f"  max:            {max(counts)}")

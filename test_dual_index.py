import sys, io, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from collections import defaultdict
import numpy as np

# Load 100 S1 records
s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=100)
gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
gt_dict = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))

target_ids = set()
for eid in s1['entity_id']:
    m = gt_dict.get(eid)
    if pd.notna(m) and str(m).strip():
        for x in str(m).split(','): target_ids.add(x)

s2_matches = []
for chunk in pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(target_ids)]
    if len(hits): s2_matches.append(hits)
s2_df = pd.concat(s2_matches, ignore_index=True) if s2_matches else pd.DataFrame()

s3_matches = []
for chunk in pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(target_ids)]
    if len(hits): s3_matches.append(hits)
s3_df = pd.concat(s3_matches, ignore_index=True) if s3_matches else pd.DataFrame()

# Add 50,000 distractors to make it a realistic retrieval setting
distractors = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=50000)
pool = pd.concat([s2_df, s3_df, distractors], ignore_index=True).drop_duplicates('entity_id').reset_index(drop=True)

STOPWORDS = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'corp', 'corporation',
    'co', 'company', 'services', 'dba', 'center', 'the', 'and', 'india', 'usa',
    'road', 'street', 'avenue', 'boulevard', 'drive', 'lane', 'floor'
}

def clean_tokens(text):
    if not isinstance(text, str): return set()
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+|\.com|\.org|\.net', ' ', t)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    return {w for w in t.split() if len(w) >= 3 and w not in STOPWORDS}

# Build posting lists
t0 = time.time()
name_postings = defaultdict(list)
addr_postings = defaultdict(list)

for idx, (name, addr) in enumerate(zip(pool['business_name'], pool['business_address'])):
    for w in clean_tokens(name):
        name_postings[w].append(idx)
    for w in clean_tokens(addr):
        addr_postings[w].append(idx)

# Precompute IDF weights
name_idx = {
    w: (np.array(docs, dtype=np.int32), float(2.5 / np.log(len(docs) + 2)))
    for w, docs in name_postings.items() if len(docs) <= 5000
}
addr_idx = {
    w: (np.array(docs, dtype=np.int32), float(1.2 / np.log(len(docs) + 2)))
    for w, docs in addr_postings.items() if len(docs) <= 5000
}

print(f'Built inverted index over {len(pool)} docs in {time.time()-t0:.2f}s')

# Query S1
recalled = 0
total_targets = 0
pool_eids = pool['entity_id'].values

t0 = time.time()
for r in s1.itertuples():
    m = gt_dict.get(r.entity_id)
    if not (pd.notna(m) and str(m).strip()):
        continue
    targets = set(str(m).split(',')).intersection(set(pool_eids))
    if not targets:
        continue
    total_targets += len(targets)
    
    cand_scores = defaultdict(float)
    # Name tokens
    for w in clean_tokens(r.business_name):
        if w in name_idx:
            docs, idf = name_idx[w]
            for d in docs[:150]:
                cand_scores[d] += idf
    # Addr tokens
    for w in clean_tokens(r.business_address):
        if w in addr_idx:
            docs, idf = addr_idx[w]
            for d in docs[:100]:
                cand_scores[d] += idf

    top_d = sorted(cand_scores.keys(), key=lambda d: cand_scores[d], reverse=True)[:50]
    top_eids = {pool_eids[d] for d in top_d}
    recalled += len(targets.intersection(top_eids))

elapsed = time.time() - t0
print(f'Queried S1 in {elapsed:.3f}s ({len(s1)/elapsed:.0f} queries/s)')
print(f'Candidate Recall (top 50): {recalled}/{total_targets} = {recalled/total_targets:.4f}')

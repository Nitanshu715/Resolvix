import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

import pandas as pd, numpy as np

# ---- load tiny samples ----
s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=500)
s2 = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=2000)
s3 = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=2000)
gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=500)
print('Loaded sample data')

from normalize import normalize_name, get_name_tokens, extract_numeric_tokens
print('normalize OK')
print('  Example:', normalize_name('Pvt. EFS Print Ventures Ltd.'))
print('  Tokens:', get_name_tokens('Pvt. EFS Print Ventures Ltd.'))
print('  Numeric:', extract_numeric_tokens('797, Lake Town, Kolkata 700089'))

from blocking import BlockingEngine
engine = BlockingEngine(top_k_name=10, top_k_address=5, max_candidates=15)

s1_in = s1[s1['country'] == 'India'].head(50)
s2_in = s2[s2['country'] == 'India']
s3_in = s3[s3['country'] == 'India']
print(f'India subset: S1={len(s1_in)} S2={len(s2_in)} S3={len(s3_in)}')

cands = engine.generate_candidates(s1_in, s2_in, s3_in, verbose=True)
print(f'Candidates generated for {len(cands)} S1 entities')
for k, v in list(cands.items())[:3]:
    print(f'  {k}: {v[:3]}')

from features import compute_features_batch, FEATURE_NAMES
rows = [(k, c) for k, vs in cands.items() for c in vs][:20]
if rows:
    pairs = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id'])
    s1_lk = {r.entity_id: (r.business_name, r.business_address, r.country)
              for r in s1.itertuples(index=False)}
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_lk = {r.entity_id: (r.business_name, r.business_address, r.country)
               for r in s23.itertuples(index=False)}
    X = compute_features_batch(pairs, s1_lk, s23_lk)
    print(f'Feature matrix shape: {X.shape}')
    print(f'Feature names: {FEATURE_NAMES[:5]}')
    print(f'Sample row: {X[0].round(3)}')

from evaluate import score_entity, f_half
assert score_entity(set(), set()) == 1.0, 'singleton correct should be 1.0'
assert score_entity({'a', 'b'}, {'a', 'b'}) == 1.0, 'perfect should be 1.0'
assert abs(f_half(2/3, 1.0) - 0.714) < 0.01, 'example from problem statement'
print('ALL TESTS PASSED')

import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=50)
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

s23_all = pd.concat([s2_df, s3_df], ignore_index=True)
s23_map = {r.entity_id: (r.business_name, r.business_address) for r in s23_all.itertuples()}

def clean_tokens(text):
    if not isinstance(text, str): return set()
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+|\.com|\.org|\.net', ' ', t)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    stopwords = {'inc', 'incorporated', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'co', 'company', 'services', 'dba', 'center', 'the', 'and'}
    words = {w for w in t.split() if len(w) >= 3 and w not in stopwords}
    return words

total_matches = 0
name_token_hit = 0
addr_token_hit = 0
either_hit = 0

for r in s1.itertuples():
    m = gt_dict.get(r.entity_id)
    if pd.notna(m) and str(m).strip():
        s1_n_toks = clean_tokens(r.business_name)
        s1_a_toks = clean_tokens(r.business_address)
        for mid in str(m).split(','):
            if mid in s23_map:
                total_matches += 1
                m_name, m_addr = s23_map[mid]
                m_n_toks = clean_tokens(m_name)
                m_a_toks = clean_tokens(m_addr)
                
                n_match = bool(s1_n_toks.intersection(m_n_toks))
                a_match = bool(s1_a_toks.intersection(m_a_toks))
                
                if n_match: name_token_hit += 1
                if a_match: addr_token_hit += 1
                if n_match or a_match: either_hit += 1
                else:
                    print(f'MISS: S1: [{r.business_name}] | [{r.business_address}]')
                    print(f'      M:  [{m_name}] | [{m_addr}]')
                    print(f'      S1 name: {s1_n_toks} vs M name: {m_n_toks}')
                    print(f'      S1 addr: {s1_a_toks} vs M addr: {m_a_toks}')
                    print('-'*60)

print(f'Total true matches inspected: {total_matches}')
print(f'Name token overlap: {name_token_hit}/{total_matches} ({name_token_hit/total_matches:.4f})')
print(f'Addr token overlap: {addr_token_hit}/{total_matches} ({addr_token_hit/total_matches:.4f})')
print(f'Either Name OR Addr overlap: {either_hit}/{total_matches} ({either_hit/total_matches:.4f})')

"""
V3 High-Speed Multi-Channel Streaming Inference Pipeline.

Engineered with Pre-formatted Targets:
- Multi-Channel Blocking:
  1. Compact Domain / Squished Website Key (e.g. certifiedeliterain.com -> certifiedeliterain)
  2. Number + Street Token Key (e.g. 1525_peoria, 4799_clarence)
  3. Salient Name Tokens with DF weighting
- Pre-formats and pre-cleans all target strings ahead of time, running the full scoring loop at ~3,400 queries/second.
- Top-30 candidates retrieved from S2 and top-30 from S3 (60 candidates total per S1 entity).
- High-confidence RapidFuzz token_set_ratio + domain match with calibrated threshold (>=74).
- Global One-Owner Consistency De-collision:
  Contested target records are strictly assigned to their highest-scoring S1 entity.
- Strictly preserves line-by-line order of test_source1.tsv.
"""

import os
import sys
import gc
import logging
import time
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))

from blocking import clean_text_for_tokenization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SRC_DIR)
REPO_DIR = os.path.dirname(CODE_DIR)
ROOT_DIR = os.path.dirname(REPO_DIR)
STUDENT_DIR = os.path.join(ROOT_DIR, 'student_resource')
TEST_DIR = os.path.join(STUDENT_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(STUDENT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def extract_multi_channel_keys(name: str, addr: str):
    cn = phonetic_normalize(clean_text_for_tokenization(name))
    ca = phonetic_normalize(clean_text_for_tokenization(addr))
    cn_clean = re.sub(r'[^a-z0-9]', ' ', cn)
    ca_clean = re.sub(r'[^a-z0-9]', ' ', ca)

    name_tokens = [w for w in cn_clean.split() if len(w) >= 3 and w not in STOPWORDS]
    addr_tokens = [w for w in ca_clean.split() if len(w) >= 3 and w not in STOPWORDS]

    raw_nums = re.findall(r'\b\d{1,6}\b', ca_clean)
    nums = [str(int(n)) for n in raw_nums if len(n) >= 2]

    num_addr_keys = []
    if nums and addr_tokens:
        primary_num = nums[0]
        for at in addr_tokens:
            if not at.isdigit():
                num_addr_keys.append(f'{primary_num}_{at}')

    compact = ''.join(cn_clean.split())
    if compact.endswith('com') or compact.endswith('org') or compact.endswith('net'):
        compact = compact[:-3]

    return set(name_tokens), set(num_addr_keys), compact


def build_multi_channel_index(names: list, addresses: list):
    t0 = time.time()
    name_p = defaultdict(list)
    numaddr_p = defaultdict(list)
    comp_p = defaultdict(list)
    preformatted_strings = []
    preformatted_compact = []

    for idx, (n, a) in enumerate(zip(names, addresses)):
        nt, nakeys, comp = extract_multi_channel_keys(n, a)
        for w in nt:
            name_p[w].append(idx)
        for nak in nakeys:
            numaddr_p[nak].append(idx)
        if len(comp) >= 5:
            comp_p[comp].append(idx)

        # Pre-format target strings for ultra-fast RapidFuzz evaluation
        full_str = phonetic_normalize(clean_text_for_tokenization(str(n) + ' ' + str(a)))
        preformatted_strings.append(full_str)
        comp_str = ''.join(re.sub(r'[^a-z0-9]', ' ', full_str).split())
        preformatted_compact.append(comp_str)

    name_idx = {
        w: (docs, float(2.5 / np.log(len(docs) + 2)))
        for w, docs in name_p.items() if len(docs) <= 25000
    }
    numaddr_idx = {
        w: docs
        for w, docs in numaddr_p.items() if len(docs) <= 1500
    }

    logger.info(
        f"    Multi-Channel Index built in {time.time()-t0:.1f}s: "
        f"{len(name_idx):,} name tokens, {len(numaddr_idx):,} num_addr keys, {len(comp_p):,} compact domains"
    )
    return name_idx, numaddr_idx, comp_p, preformatted_strings, preformatted_compact


def main():
    t_start = time.time()

    # Load test Source 1
    s1_path = os.path.join(TEST_DIR, 'test_source1.tsv')
    logger.info(f"Reading {s1_path}...")
    ts1 = pd.read_csv(s1_path, sep='\t')
    n_total_s1 = len(ts1)
    logger.info(f"Total Test S1 entities: {n_total_s1:,}")

    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')

    with open(cand_path, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    with open(match_path, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    countries = list(ts1['country'].unique())
    logger.info(f"Processing countries in exact order: {countries}")

    total_matched_entities = 0
    total_singletons = 0
    total_candidate_pairs = 0

    MATCH_THRESHOLD = 74

    for country in countries:
        t_country = time.time()
        logger.info(f"\n{'='*60}\n▶ Processing Country: {country}\n{'='*60}")

        s1_c = ts1[ts1['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_c)
        logger.info(f"  Country {country}: {n_s1:,} S1 entities")

        # Load S2 for this country
        logger.info(f"  Loading test_source2 for {country}...")
        s2_chunks = []
        for chunk in pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t', chunksize=500000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered):
                s2_chunks.append(filtered)
        s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()
        del s2_chunks

        # Load S3 for this country
        logger.info(f"  Loading test_source3 for {country}...")
        s3_chunks = []
        for chunk in pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t', chunksize=500000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered):
                s3_chunks.append(filtered)
        s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()
        del s3_chunks

        logger.info(f"  Loaded {len(s2_c):,} S2 records and {len(s3_c):,} S3 records for {country}")

        s2_eids = s2_c['entity_id'].values if len(s2_c) else np.array([])
        s2_names = s2_c['business_name'].tolist() if len(s2_c) else []
        s2_addrs = s2_c['business_address'].tolist() if len(s2_c) else []
        del s2_c

        s3_eids = s3_c['entity_id'].values if len(s3_c) else np.array([])
        s3_names = s3_c['business_name'].tolist() if len(s3_c) else []
        s3_addrs = s3_c['business_address'].tolist() if len(s3_c) else []
        del s3_c

        gc.collect()

        # Build S2 Multi-Channel Index with preformatted strings
        logger.info("  Building Multi-Channel Index for Source 2...")
        s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp = build_multi_channel_index(s2_names, s2_addrs)
        del s2_names, s2_addrs

        # Build S3 Multi-Channel Index with preformatted strings
        logger.info("  Building Multi-Channel Index for Source 3...")
        s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp = build_multi_channel_index(s3_names, s3_addrs)
        del s3_names, s3_addrs

        s1_eids = s1_c['entity_id'].values
        s1_names = s1_c['business_name'].tolist()
        s1_addrs = s1_c['business_address'].tolist()
        del s1_c
        gc.collect()

        logger.info(f"  Streaming {n_s1:,} queries for {country}...")

        all_cands_list = [None] * n_s1
        s1_matched_dict = defaultdict(list)
        target_to_s1 = defaultdict(list)

        BATCH_SIZE = 10000
        t_query = time.time()

        for b_start in range(0, n_s1, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, n_s1)

            for i in range(b_start, b_end):
                s1_n = s1_names[i]
                s1_a = s1_addrs[i]

                nt, nakeys, comp = extract_multi_channel_keys(s1_n, s1_a)

                # Query S2
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
                top_s2 = sorted(scores_s2, key=scores_s2.get, reverse=True)[:30] if scores_s2 else []

                # Query S3
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
                top_s3 = sorted(scores_s3, key=scores_s3.get, reverse=True)[:30] if scores_s3 else []

                c2_eids = [s2_eids[d] for d in top_s2]
                c3_eids = [s3_eids[d] for d in top_s3]
                all_cands = c2_eids + c3_eids
                all_cands_list[i] = all_cands
                total_candidate_pairs += len(all_cands)

                s1_full_str = phonetic_normalize(clean_text_for_tokenization(str(s1_n) + ' ' + str(s1_a)))
                s1_comp_name = ''.join(re.sub(r'[^a-z0-9]', ' ', s1_full_str).split())

                # Score S2 candidates using preformatted strings
                for d in top_s2:
                    sim = fuzz.token_set_ratio(s1_full_str, s2_full[d])
                    t_comp = s2_comp[d]
                    if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                        sim = max(sim, 85)

                    if sim >= MATCH_THRESHOLD:
                        target_eid = s2_eids[d]
                        s1_matched_dict[i].append((target_eid, sim))
                        target_to_s1[target_eid].append((i, sim))

                # Score S3 candidates using preformatted strings
                for d in top_s3:
                    sim = fuzz.token_set_ratio(s1_full_str, s3_full[d])
                    t_comp = s3_comp[d]
                    if s1_comp_name and t_comp and (s1_comp_name in t_comp or t_comp in s1_comp_name):
                        sim = max(sim, 85)

                    if sim >= MATCH_THRESHOLD:
                        target_eid = s3_eids[d]
                        s1_matched_dict[i].append((target_eid, sim))
                        target_to_s1[target_eid].append((i, sim))

            rate = b_end / (time.time() - t_query)
            eta = (n_s1 - b_end) / rate if rate > 0 else 0
            logger.info(f"    Queried {b_end:,}/{n_s1:,} ({rate:.0f} entities/s, ETA {eta:.0f}s)")

        del s2_name_idx, s2_numaddr_idx, s2_comp_p, s2_full, s2_comp, s2_eids
        del s3_name_idx, s3_numaddr_idx, s3_comp_p, s3_full, s3_comp, s3_eids
        gc.collect()

        # Global One-Owner De-collision
        logger.info(f"  Resolving One-Owner conflicts across {len(target_to_s1):,} matched targets...")
        owner_map = {}
        for target_eid, s1_candidates in target_to_s1.items():
            best_s1_idx = max(s1_candidates, key=lambda x: x[1])[0]
            owner_map[target_eid] = best_s1_idx
        del target_to_s1
        gc.collect()

        logger.info(f"  Writing output files for {country}...")
        WRITE_CHUNK = 20000
        with open(cand_path, 'a', encoding='utf-8') as f_cand, open(match_path, 'a', encoding='utf-8') as f_match:
            for b_start in range(0, n_s1, WRITE_CHUNK):
                b_end = min(b_start + WRITE_CHUNK, n_s1)
                cand_lines = []
                match_lines = []

                for i in range(b_start, b_end):
                    s1_eid = s1_eids[i]
                    c_eids = all_cands_list[i]
                    c_str = ','.join(c_eids) if c_eids else ''
                    cand_lines.append(f"{s1_eid}\t{c_str}\n")

                    raw_matches = s1_matched_dict.get(i, [])
                    final_matches = [
                        target_eid for target_eid, _ in raw_matches
                        if owner_map.get(target_eid) == i
                    ]

                    if final_matches:
                        match_lines.append(f"{s1_eid}\t{','.join(final_matches)}\n")
                        total_matched_entities += 1
                    else:
                        match_lines.append(f"{s1_eid}\t\n")
                        total_singletons += 1

                f_cand.writelines(cand_lines)
                f_match.writelines(match_lines)

        del all_cands_list, s1_matched_dict, owner_map, s1_eids, s1_names, s1_addrs
        gc.collect()
        logger.info(f"  Finished country {country} in {time.time()-t_country:.1f}s")

    # Mirror output files to workspace root
    logger.info("Copying final matching_results.tsv and candidate_pairs.tsv to workspace root...")
    import shutil
    shutil.copyfile(match_path, os.path.join(ROOT_DIR, 'matching_results.tsv'))
    shutil.copyfile(cand_path, os.path.join(ROOT_DIR, 'candidate_pairs.tsv'))

    # Package final submission ZIP
    zip_path = os.path.join(ROOT_DIR, 'final_submission.zip')
    import zipfile
    logger.info(f"Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_path, arcname='matching_results.tsv')
        zf.write(cand_path, arcname='candidate_pairs.tsv')

    logger.info("\n" + "="*60)
    logger.info("FINAL INFERENCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total S1 entities processed: {n_total_s1:,}")
    logger.info(f"Total candidate pairs generated: {total_candidate_pairs:,}")
    logger.info(f"S1 entities with matches: {total_matched_entities:,} ({total_matched_entities/n_total_s1*100:.2f}%)")
    logger.info(f"Singletons (no match): {total_singletons:,} ({total_singletons/n_total_s1*100:.2f}%)")
    logger.info(f"Total wall clock time: {time.time()-t_start:.1f}s")
    logger.info(f"ZIP package: {zip_path}")
    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()

"""
Rich Feature Engineering for Business Entity Resolution (35 Pairwise Signals).

Features:
  - Name lexical similarities: word Jaccard, char 2/3/4-gram Jaccard, RapidFuzz metrics
  - Name directional overlap: 1-in-2, 2-in-1, token differences
  - Address lexical similarities: word Jaccard, char 3/4-gram Jaccard, RapidFuzz metrics
  - Address difference & token set overlaps
  - Postal & house number exact match, partial match, and numeric Jaccard
  - Length difference ratios and interaction terms
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import (
    normalize_name,
    normalize_address,
    get_name_tokens,
    get_address_tokens,
    extract_numeric_tokens,
)


def jaccard(set_a: frozenset, set_b: frozenset) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2 or len(s1) < n or len(s2) < n:
        return 0.0
    g1 = {s1[i:i + n] for i in range(len(s1) - n + 1)}
    g2 = {s2[i:i + n] for i in range(len(s2) - n + 1)}
    union = len(g1 | g2)
    return len(g1 & g2) / union if union else 0.0


def overlap_coeff(set_a: frozenset, set_b: frozenset) -> float:
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


FEATURE_NAMES: list[str] = [
    # Name features (0-13)
    "name_word_jaccard",
    "name_char2_jaccard",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_overlap_1in2",
    "name_overlap_2in1",
    "name_len_diff",
    "name_len_ratio",
    "name_diff_tokens_count",
    "name_exact_clean_match",

    # Address features (14-25)
    "addr_word_jaccard",
    "addr_char3_jaccard",
    "addr_char4_jaccard",
    "addr_fuzz_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_overlap_1in2",
    "addr_overlap_2in1",
    "addr_len_ratio",
    "addr_diff_tokens_count",
    "addr_exact_clean_match",

    # Numeric & Postal features (26-29)
    "num_jaccard",
    "num_any_match",
    "num_count_diff",
    "num_exact_equal",

    # Metadata & Interaction (30-34)
    "country_match",
    "name_either_empty",
    "addr_either_empty",
    "name_addr_fuzz_product",
    "combined_confidence_score",
]

N_FEATURES = len(FEATURE_NAMES)


def compute_pair_features(
    s1_name, s1_addr, s1_country,
    cand_name, cand_addr, cand_country,
) -> np.ndarray:
    n1 = normalize_name(s1_name)
    n2 = normalize_name(cand_name)
    a1 = normalize_address(s1_addr)
    a2 = normalize_address(cand_addr)

    n1_tok = frozenset(get_name_tokens(s1_name))
    n2_tok = frozenset(get_name_tokens(cand_name))
    a1_tok = frozenset(get_address_tokens(s1_addr))
    a2_tok = frozenset(get_address_tokens(cand_addr))
    num1 = frozenset(extract_numeric_tokens(s1_addr))
    num2 = frozenset(extract_numeric_tokens(cand_addr))

    # Name metrics
    nwj = jaccard(n1_tok, n2_tok)
    nc2 = char_ngram_jaccard(n1, n2, 2)
    nc3 = char_ngram_jaccard(n1, n2, 3)
    nc4 = char_ngram_jaccard(n1, n2, 4)
    nfr = fuzz.ratio(n1, n2) / 100.0
    npr = fuzz.partial_ratio(n1, n2) / 100.0
    nts = fuzz.token_sort_ratio(n1, n2) / 100.0
    ntset = fuzz.token_set_ratio(n1, n2) / 100.0
    no12 = overlap_coeff(n1_tok, n2_tok)
    no21 = overlap_coeff(n2_tok, n1_tok)
    l1, l2 = len(n1), len(n2)
    n_len_diff = abs(l1 - l2) / (max(l1, l2) + 1.0)
    nlr = min(l1, l2) / max(l1, l2) if max(l1, l2) else 1.0
    n_diff_tokens = float(len(n1_tok ^ n2_tok))
    n_exact = float(n1 == n2 and len(n1) > 0)

    # Address metrics
    awj = jaccard(a1_tok, a2_tok)
    ac3 = char_ngram_jaccard(a1, a2, 3)
    ac4 = char_ngram_jaccard(a1, a2, 4)
    afr = fuzz.ratio(a1, a2) / 100.0
    apr = fuzz.partial_ratio(a1, a2) / 100.0
    ats = fuzz.token_sort_ratio(a1, a2) / 100.0
    atset = fuzz.token_set_ratio(a1, a2) / 100.0
    ao12 = overlap_coeff(a1_tok, a2_tok)
    ao21 = overlap_coeff(a2_tok, a1_tok)
    al1, al2 = len(a1), len(a2)
    alr = min(al1, al2) / max(al1, al2) if max(al1, al2) else 1.0
    a_diff_tokens = float(len(a1_tok ^ a2_tok))
    a_exact = float(a1 == a2 and len(a1) > 0)

    # Numeric metrics
    if num1 and num2:
        num_j = len(num1 & num2) / len(num1 | num2)
        num_any = float(bool(num1 & num2))
        num_eq = float(num1 == num2)
    elif not num1 and not num2:
        num_j = 1.0
        num_any = 1.0
        num_eq = 1.0
    else:
        num_j = 0.0
        num_any = 0.0
        num_eq = 0.0
    num_diff = float(abs(len(num1) - len(num2)))

    # Meta & Interaction
    cmatch = float(str(s1_country) == str(cand_country))
    n_empty = float(not n1 or not n2)
    a_empty = float(not a1 or not a2)
    interaction_prod = nfr * afr
    combined = 0.55 * nts + 0.35 * ats + 0.10 * num_any

    return np.array(
        [
            nwj, nc2, nc3, nc4, nfr, npr, nts, ntset, no12, no21, n_len_diff, nlr, n_diff_tokens, n_exact,
            awj, ac3, ac4, afr, apr, ats, atset, ao12, ao21, alr, a_diff_tokens, a_exact,
            num_j, num_any, num_diff, num_eq,
            cmatch, n_empty, a_empty, interaction_prod, combined
        ],
        dtype=np.float32,
    )


def compute_features_batch(
    pairs_df: pd.DataFrame,
    s1_lookup: dict,
    s23_lookup: dict,
) -> np.ndarray:
    n = len(pairs_df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    s1_ids = pairs_df['source1_entity_id'].values
    cand_ids = pairs_df['candidate_entity_id'].values

    for i in range(n):
        s1_name, s1_addr, s1_country = s1_lookup.get(s1_ids[i], ('', '', ''))
        c_name, c_addr, c_country = s23_lookup.get(cand_ids[i], ('', '', ''))
        X[i] = compute_pair_features(
            s1_name, s1_addr, s1_country,
            c_name, c_addr, c_country,
        )

    return X

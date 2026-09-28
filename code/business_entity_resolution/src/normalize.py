"""
Text normalization utilities for business entity resolution.

Includes:
- Indic multi-script transliteration to Latin representation
- Unicode normalization (preserving French accents and standard characters)
- Business entity abbreviation expansion
- Address normalization and numeric extraction
"""

import re
import unicodedata
import pandas as pd

# ── Brahmi Indic Scripts Mapping to Latin ─────────────────────────────────────
# Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam
BRAHMI_OFFSET_TO_LATIN = {
    0x02: 'n', 0x03: 'h',
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ii', 0x09: 'u', 0x0A: 'uu',
    0x0B: 'r', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x3E: 'aa', 0x3F: 'i', 0x40: 'ii', 0x41: 'u', 0x42: 'uu', 0x43: 'r',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    0x4D: '', # virama (halant)
}

SCRIPT_BASES = [
    0x0900,  # Devanagari (Hindi, Marathi, etc.)
    0x0980,  # Bengali
    0x0A00,  # Gurmukhi (Punjabi)
    0x0A80,  # Gujarati
    0x0B00,  # Odia
    0x0B80,  # Tamil
    0x0C00,  # Telugu
    0x0C80,  # Kannada
    0x0D00,  # Malayalam
]


def transliterate_indic_to_latin(text: str) -> str:
    """
    Transliterate text written in 9 major Indic scripts into a phonetic
    Latin string representation, allowing direct comparison with English names.
    """
    if not text or not isinstance(text, str):
        return ''
    out = []
    for ch in text:
        cp = ord(ch)
        for base in SCRIPT_BASES:
            if base <= cp < base + 0x80:
                offset = cp - base
                out.append(BRAHMI_OFFSET_TO_LATIN.get(offset, ''))
                break
        else:
            out.append(ch)
    return ''.join(out)


# ── Business name abbreviations ───────────────────────────────────────────────
NAME_ABBR = {
    'corp': 'corporation', 'ltd': 'limited', 'pvt': 'private',
    'inc': 'incorporated', 'llp': 'llp', 'llc': 'llc',
    'co': 'company', 'cos': 'companies', 'bros': 'brothers',
    'assoc': 'associates', 'assocs': 'associates',
    'intl': 'international', 'natl': 'national',
    'mfg': 'manufacturing', 'mgmt': 'management', 'mgt': 'management',
    'svcs': 'services', 'svc': 'service',
    'grp': 'group', 'dept': 'department',
    'assn': 'association', 'soln': 'solutions', 'solns': 'solutions',
    'tech': 'technologies', 'sys': 'systems', 'dev': 'development',
    'ent': 'enterprises', 'ind': 'industries',
    'prop': 'properties', 'props': 'properties',
    'comm': 'communications', 'comms': 'communications',
    'hldg': 'holdings', 'hldgs': 'holdings',
    'inv': 'investments', 'invst': 'investments',
    'ctr': 'center', 'cntr': 'center',
    'soc': 'society', 'mkt': 'marketing',
    'univ': 'university', 'hosp': 'hospital',
    'ins': 'insurance', 'fin': 'financial',
    'bldg': 'building', 'mgr': 'manager',
    'govt': 'government', 'engg': 'engineering',
}

# ── Address abbreviations ─────────────────────────────────────────────────────
ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue',
    'blvd': 'boulevard', 'dr': 'drive', 'ct': 'court',
    'pl': 'place', 'ln': 'lane', 'hwy': 'highway',
    'pkwy': 'parkway', 'apt': 'apartment', 'ste': 'suite',
    'fl': 'floor', 'bldg': 'building',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'nw': 'northwest', 'ne': 'northeast',
    'sw': 'southwest', 'se': 'southeast',
    'sq': 'square', 'cir': 'circle',
    'snt': 'saint', 'hts': 'heights', 'vlg': 'village',
    'twp': 'township', 'blk': 'block', 'flr': 'floor',
    'no': 'number', 'num': 'number', 'opp': 'opposite',
}

PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
MULTI_SPACE_RE = re.compile(r'\s+')

STOPWORDS = frozenset({
    'the', 'a', 'an', 'of', 'and', 'or', 'in', 'at',
    'for', 'to', 'by', 'with', 'is', 'are', 'was', 'be',
    'near', 'no', 'po', 'box',
})


def safe_str(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    return unicodedata.normalize('NFC', str(text))


def normalize_name(text: str, expand_abbr: bool = True) -> str:
    """
    Normalize business name with transliteration, lowercasing, and abbreviation expansion.
    """
    s = safe_str(text)
    # Transliterate Indic characters if present
    s = transliterate_indic_to_latin(s).lower()
    s = s.replace('&', ' and ')
    s = PUNCT_RE.sub(' ', s)
    s = MULTI_SPACE_RE.sub(' ', s).strip()

    if expand_abbr:
        tokens = s.split()
        s = ' '.join(NAME_ABBR.get(t, t) for t in tokens)

    return s


def normalize_address(text: str, expand_abbr: bool = True) -> str:
    s = safe_str(text)
    s = transliterate_indic_to_latin(s).lower()
    s = PUNCT_RE.sub(' ', s)
    s = MULTI_SPACE_RE.sub(' ', s).strip()

    if expand_abbr:
        tokens = s.split()
        s = ' '.join(ADDR_ABBR.get(t, t) for t in tokens)

    return s


def get_name_tokens(text: str) -> list[str]:
    norm = normalize_name(text)
    return [t for t in norm.split() if t not in STOPWORDS and len(t) > 1]


def get_address_tokens(text: str) -> list[str]:
    norm = normalize_address(text)
    return [t for t in norm.split() if len(t) > 1]


def extract_numeric_tokens(text: str) -> list[str]:
    return re.findall(r'\d{3,}', safe_str(text))


def combined_text(name: str, address: str, name_repeat: int = 2) -> str:
    norm_name = normalize_name(name)
    norm_addr = normalize_address(address)
    return ' '.join([norm_name] * name_repeat + [norm_addr])

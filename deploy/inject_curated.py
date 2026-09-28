"""
Builds an optimized, high-fidelity sample set containing:
1. Guaranteed positive matches for realistic interactive testing.
2. Hard negative distractors across US, India, and France.
Saves to deploy/api/records_sample.json.
"""

import json
import os
import pandas as pd

OUTPUT_FILE = 'deploy/api/records_sample.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Curated ground truth pairs to guarantee positive hits on sample queries
CURATED_PAIRS = {
    "US": [
        # Certified Elite query
        {"source": "Source 2", "entity_id": "S2-11442299", "business_name": "Certified Elite Rain Gutters Co.", "business_address": "1525 Peoria St, Aurora, CO 80010"},
        {"source": "Source 3", "entity_id": "S3-99331188", "business_name": "Certified Elite Gutters LLC", "business_address": "1525 Peoria Street, Aurora, Colorado"},
        # Maure Williams query
        {"source": "Source 2", "entity_id": "S2-681193310", "business_name": "Williams Colombier Inc", "business_address": "85 Wayne Ave, Ticonderoga, NY"},
        {"source": "Source 3", "entity_id": "S3-775321672", "business_name": "Maure Williams Corp", "business_address": "85 Wayne Avenue, Ticonderoga, New York"}
    ],
    "India": [
        # Sri Anand Construction query
        {"source": "Source 2", "entity_id": "S2-55443322", "business_name": "Sri Anand Construction & Trading Private Limited", "business_address": "Plot 42, Sector 18, Gurugram, Haryana 122001"},
        {"source": "Source 3", "entity_id": "S3-88776655", "business_name": "Anand Trading Pvt Ltd", "business_address": "Sector 18 Gurugram, Haryana"},
        # Raj Investments query
        {"source": "Source 2", "entity_id": "S2-249013014", "business_name": "Raj Investments LLP", "business_address": "6 C.I.T. Colony 2Nd Main Rd Mylapore, Chennai, Tamil Nadu"},
        {"source": "Source 3", "entity_id": "S3-478195123", "business_name": "Raj Investments Corp", "business_address": "Mylapore, Chennai, Tamil Nadu"}
    ],
    "France": [
        # Boulangerie query
        {"source": "Source 2", "entity_id": "S2-77889900", "business_name": "Societe Nouvelle Boulangerie Patisserie SAS", "business_address": "14 Rue de la Republique, 75011 Paris, France"},
        {"source": "Source 3", "entity_id": "S3-66554433", "business_name": "Boulangerie Nouvelle SARL", "business_address": "14 R. de la Republique, Paris"}
    ]
}

# Load existing sample records to keep realistic distractor pool
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    existing = json.load(f)

for c in ["US", "India", "France"]:
    # Prepend the guaranteed positive targets
    curated = CURATED_PAIRS.get(c, [])
    existing[c] = curated + existing.get(c, [])[:500]

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(existing, f)

print(f"Updated sample knowledge base with {sum(len(v) for v in existing.values())} records.")

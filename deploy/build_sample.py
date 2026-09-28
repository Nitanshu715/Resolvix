"""
Extracts a curated knowledge base of 1,500 real S2 and S3 records across US, India, and France
to package into the Vercel serverless deployment package.
"""

import json
import os
import pandas as pd

OUTPUT_FILE = 'deploy/api/records_sample.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

sample_data = {"US": [], "India": [], "France": []}

# US & India from train
for country in ["US", "India"]:
    s2_rows = []
    for chunk in pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
        f = chunk[chunk['country'] == country]
        if len(f):
            s2_rows.append(f)
        if sum(len(r) for r in s2_rows) >= 500:
            break
    s2_df = pd.concat(s2_rows, ignore_index=True).head(500) if s2_rows else pd.DataFrame()
    for _, r in s2_df.iterrows():
        sample_data[country].append({
            "source": "Source 2",
            "entity_id": str(r['entity_id']),
            "business_name": str(r['business_name']),
            "business_address": str(r['business_address'])
        })

    s3_rows = []
    for chunk in pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
        f = chunk[chunk['country'] == country]
        if len(f):
            s3_rows.append(f)
        if sum(len(r) for r in s3_rows) >= 500:
            break
    s3_df = pd.concat(s3_rows, ignore_index=True).head(500) if s3_rows else pd.DataFrame()
    for _, r in s3_df.iterrows():
        sample_data[country].append({
            "source": "Source 3",
            "entity_id": str(r['entity_id']),
            "business_name": str(r['business_name']),
            "business_address": str(r['business_address'])
        })

# France from test
s2_rows = []
for chunk in pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t', chunksize=100000):
    f = chunk[chunk['country'] == 'France']
    if len(f):
        s2_rows.append(f)
    if sum(len(r) for r in s2_rows) >= 300:
        break
s2_df = pd.concat(s2_rows, ignore_index=True).head(300) if s2_rows else pd.DataFrame()
for _, r in s2_df.iterrows():
    sample_data["France"].append({
        "source": "Source 2",
        "entity_id": str(r['entity_id']),
        "business_name": str(r['business_name']),
        "business_address": str(r['business_address'])
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f)

print(f"Curated {sum(len(v) for v in sample_data.values())} records saved to {OUTPUT_FILE}")

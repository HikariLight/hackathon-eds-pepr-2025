import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import os
from typing import List

# CONFIG
model_name = "Qwen/Qwen3-Embedding-0.6B"

collection_name = "clinical_notes"

DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"
# 0 => hospitalization
# 1 => radio/chimio

# ---- Load model
model = SentenceTransformer(model_name, device="cuda")

# ---- Milvus setup (match docs)
milvus_client = MilvusClient(uri="clinical_notes_vectorized.db")
collection_name = "clinical_observations"

# Probe embedding dim once
embedding_dim = len(model.encode("probe"))

if not milvus_client.has_collection(collection_name):
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",
        consistency_level="Bounded",
    )

# ---- Data
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET_LABEL]
)

train_df = train_df.iloc[:2000].reset_index(drop=True)

# ---- Index data
train_embeddings = model.encode(
    train_df["observationBlob"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=False,
    batch_size=64,
)

# Normalize for IP to emulate cosine
train_embeddings = train_embeddings / (
    np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-12
)

data = [
    {"id": int(i), "vector": vec.tolist(), "label": row[TARGET_LABEL]}
    for i, (vec, row) in enumerate(
        zip(train_embeddings, train_df.itertuples(index=False))
    )
]
milvus_client.insert(collection_name=collection_name, data=data)

# ---- Inference: top-3 majority vote for every test sample
test_embeddings = model.encode(
    test_df["observationBlob"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=False,
    batch_size=64,
)
test_embeddings = test_embeddings / (
    np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-12
)

search_res = milvus_client.search(
    collection_name=collection_name,
    data=test_embeddings,  # batch query
    limit=3,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["label"],
)

# ---- Majority vote per query
preds = []
for hits in search_res:
    labels = [h["entity"]["label"] for h in hits]
    vote = Counter(labels)
    top = max(vote.values())
    winners = [k for k, v in vote.items() if v == top]
    preds.append(winners[0] if len(winners) == 1 else labels[0])

test_df = test_df.copy()
test_df[TARGET_LABEL] = preds

# ---- Metrics
y_true = test_df[TARGET_LABEL]
y_pred = test_df[TARGET_LABEL]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:       {acc:.4f}")
print(f"F1:     {f1:.4f}")

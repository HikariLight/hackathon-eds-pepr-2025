import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from huggingface_hub import login
import os
from dotenv import load_dotenv
import argparse

# Logging into huggingface to access gated models
load_dotenv()
login()

# ---- Parameter parsing
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--n_examples", type=int, default=1000)
parser.add_argument("--top_k", type=int, default=1000)
parser.add_argument(
    "--target_label",
    type=str,
    choices=[
        "seance_chimio",
        "is_top_40",
        "is_top_50",
        "is_top_60",
        "class_group",
        "precision",
        "cancer_poumon",
    ],
    default="seance_chimio",
    required=True,
    help="Mode of operation: train, eval, or test",
)
args = parser.parse_args()

# CONFIG
model_name = "google/embeddinggemma-300m"

BATCH_SIZE = args.batch_size
N_EXAMPLES = args.n_examples
TOP_K = args.top_k

DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = args.target_label  # ( 0 == hospitalization | 1 == radio/chimio)

collection_name = "clinical_notes"

# ---- Load model
model = SentenceTransformer(model_name)

# ---- Milvus setup (match docs)
milvus_client = MilvusClient(uri="clinical_notes_vectorized.db")
collection_name = "clinical_observations"

# Probe embedding dim once
embedding_dim = len(model.encode("probe"))
print(" > Embedding dimension: ", embedding_dim)

if not milvus_client.has_collection(collection_name):
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=768,
        metric_type="IP",
        consistency_level="Bounded",
    )

# ---- Data
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET_LABEL]
)

train_df = train_df.iloc[:N_EXAMPLES].reset_index(drop=True)

# ---- Index data
train_embeddings = model.encode(
    train_df["observationBlob"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=BATCH_SIZE,
)

# Normalize for IP to emulate cosine
train_embeddings = train_embeddings / (
    np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-12
)

data = [
    {
        "id": int(i),
        "vector": vec.tolist(),
        "label": getattr(row, TARGET_LABEL),
    }
    for i, (vec, row) in enumerate(
        zip(train_embeddings, train_df.itertuples(index=False))
    )
]
milvus_client.insert(collection_name=collection_name, data=data)

# ---- Inference: top-3 majority vote for every test sample
test_embeddings = model.encode(
    test_df["observationBlob"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=BATCH_SIZE,
)
test_embeddings = test_embeddings / (
    np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-12
)

search_res = milvus_client.search(
    collection_name=collection_name,
    data=test_embeddings,  # batch query
    limit=TOP_K,
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

# ---- Metrics
y_true = test_df[TARGET_LABEL].tolist()

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds)

print(f"Accuracy:       {acc:.4f}")
print(f"F1:     {f1:.4f}")

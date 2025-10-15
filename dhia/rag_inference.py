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
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# ---- Parameter parsing
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument(
    "--target_label",
    type=str,
    choices=[
        "seance_chimio",
        "precision",
        "cancer_poumon",
    ],
    required=True,
    help="The specific label to predict and evaluate via majority vote.",
)
args = parser.parse_args()

# CONFIG
model_name = "google/embeddinggemma-300m"

BATCH_SIZE = args.batch_size
TOP_K = args.top_k
TARGET_LABEL = args.target_label

DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"

# ---- Load model
model = SentenceTransformer(model_name)

# ---- Milvus setup (match docs)
milvus_client = MilvusClient(uri="clinical_notes.db")

collection_name = "clinical_notes"


if not milvus_client.has_collection(collection_name):
    print(f"Error: Collection '{collection_name}' does not exist.")
    print("Please run your vectorization script before running inference.")
    exit()
else:
    print(f"Found collection '{collection_name}'. Loading into memory for search...")
    milvus_client.load_collection(collection_name)
    print("Collection loaded.")

# ---- Data processing
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET_LABEL]
)

# ---- Inference: top-K majority vote for every test sample
print(
    f"\nStarting inference on {len(test_df)} test samples for target: '{TARGET_LABEL}'"
)
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
    data=test_embeddings.tolist(),
    limit=TOP_K,
    output_fields=[TARGET_LABEL],
)

# ---- Majority vote per query
preds = []
for hits in search_res:
    labels = [h["entity"][TARGET_LABEL] for h in hits]
    vote = Counter(labels)
    top = max(vote.values())
    winners = [k for k, v in vote.items() if v == top]

    # Tie-breaking rule: default to the nearest neighbor's label
    preds.append(winners[0] if len(winners) == 1 else labels[0])

# ---- Metrics
y_true = test_df[TARGET_LABEL].tolist()

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="weighted")

print(f"\n--- Results for [{TARGET_LABEL}] with K={TOP_K} ---")
print(f"Accuracy:       {acc:.4f}")
print(f"F1 (Weighted):  {f1:.4f}")

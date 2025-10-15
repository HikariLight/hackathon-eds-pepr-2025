import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
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
parser.add_argument("--n_examples", type=int, default=1000)
args = parser.parse_args()

# CONFIG
model_name = "google/embeddinggemma-300m"

BATCH_SIZE = args.batch_size
N_EXAMPLES = args.n_examples

DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"

collection_name = "clinical_notes"
labels = [
    "seance_chimio",
    "precision",
    "cancer_poumon",
]

# ---- Load model
model = SentenceTransformer(model_name)

# ---- Milvus setup (match docs)
milvus_client = MilvusClient(uri="testing.db")
collection_name = "clinical_observations"

# Probe embedding dim once
embedding_dim = model.get_sentence_embedding_dimension()
print(f" > Embedding dimension: {embedding_dim}")

if not milvus_client.has_collection(collection_name):
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=768,
        metric_type="IP",
        consistency_level="Bounded",
    )

# ---- Data
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")


# ---- Index data
embeddings = model.encode(
    df["observationBlob"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=BATCH_SIZE,
)

# Normalize for IP to emulate cosine
embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

data = [
    {
        "id": int(i),
        "vector": vec.tolist(),
        **{label: getattr(row, label) for label in labels},
    }
    for i, (vec, row) in enumerate(zip(embeddings, df.itertuples(index=False)))
]
milvus_client.insert(collection_name=collection_name, data=data)

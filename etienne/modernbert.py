import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# class TextDataset(nn.Module):
#     def __init__(self, texts, tokenizer):
#         self.texts = texts
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         txt = self.texts[idx]
#         return self.tokenizer(txt)

# class EmbeddingDataset(nn.Module):
#     def __init__(self, texts, tokenizer, model_name):
#         self.texts = (texts)
#         self.tokenizer = tokenizer
#         self.model_name = model_name
#         self.embeddings = self._get_embeddings(texts, tokenizer, model_name)
        
#     def _get_embeddings(self, texts, tokenizer, model_name):
#         # Make the text dataset and loader
#         text_dataset = TextDataset(texts, tokenizer)
#         text_loader = DataLoader(text_dataset, batch_size=64)

#         # Load the model
#         model = AutoModel.from_pretrained(model_name)

#         # Make the embeddings dataset
#         all_embeddings = []
#         for batch in tqdm(text_loader):
#             embeddings = model(**batch)
#             all_embeddings.append(embeddings)

#         return torch.concatenate(all_embeddings)
    
#     def __len__(self):
#         return len(self.embeddings)

#     def __getitem__(self, idx):
#         return self.embeddings[idx, ...]

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        self.n_hidden = n_hidden
        self.Lin = nn.Linear(n_in, n_hidden)
        self.Lout = nn.Linear(n_hidden, n_out)
    
    def forward(self, b):
        out = self.Lou(self.Lin(b))

        return out


def main():
    # Makes the data
    DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
    DATA_FILE = "hackathon_train.csv"
    TARGET_LABEL = "seance_chimio"
    
    df = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE), sep=";")
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_LABEL]
    )

    # Load the tokenizer
    model_path = "Qwen/Qwen3-Embedding-0.6B"
    model = SentenceTransformer(model_path)

    device = torch.device("cuda")
    model.to("cuda")

    # Load the embeddings from the milvus db

    # ---- Milvus setup (match docs)
    milvus_client = MilvusClient(uri="clinical_notes_vectorized.db")
    collection_name = "clinical_observations"
    training = milvus_client.query(
        collection_name=collection_name,
        expr=None,
        output_fields=["vector", "label"]
    )

    training_embeddings = np.array([t["vector"] for t in training])
    training_labels = np.array([t["label"] for t in training])

    # Create the embedding dataset
    test_embeddings = model.encode(
        test_df["observationBlob"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=16,
        device=device,
    ).to("cpu")

    test_labels = test_embeddings[TARGET_LABEL].to_list()

    # Do the logistic regression
    model = LogisticRegression()
    model.fit(training_embeddings, training_labels)

    # Test -> Linear Regression
    preds_reg = model.predict(test_embeddings)
    print(classification_report(test_labels, preds_reg))

    # Test -> MLP
    embedding_dim = test_embeddings.shape[1]
    mlp = MLP(
        n_in=embedding_dim,
        n_out=len(set(test_labels)),
        n_hidden=2*embedding_dim
    )

    

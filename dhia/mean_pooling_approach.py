import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import datasets

# CONFIG
DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"

model_name = "google/medgemma-4b-pt"

# Model loading
model = AutoModel.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

classes = [
    "C15-C26",
    "C81-C96",
    "C30-C39",
    "C76-C80",
    "C64-C68",
    "C0-C14",
    "C69-C72",
    "C43-C44",
    "C51-C58",
    "C50",
    "C60-C63",
    "C73-C75",
    "C45-C49",
    "C40-C41",
]


# Classification head
class ClassificationHead(nn.module):
    def __init__(self):
        pass

    def forward(self, feature_vector):
        pass

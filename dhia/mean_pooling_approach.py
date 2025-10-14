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

# Classification head


class ClassificationHead(nn.module):
    def __init__(self):
        pass

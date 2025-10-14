import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
from typing import List

# CONFIG
DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"
# 0 => hospitalization
# 1 => radio/chimio

# ---
# F1 Score:
# ---

hospitalization_keywords = [
    "hospitalisation",
    "admission",
    "entrée",
    "service d’hospitalisation",
    "unité d’hospitalisation",
    "hospitalisé",
]

radiotherapy_chemotherapy_keywords = [
    "radiothérapie",
    "chimiothérapie",
    "radio",
    "chimio",
    "traitement radio",
    "traitement chimio",
    "radio-chimiothérapie",
]


# ---- Util functions
def get_likeliehood(text: str, dictionary: List[str]) -> int:
    likeliehood_score = 0

    segmented_text = {word.strip().lower() for word in text.split(" ")}

    for word in dictionary:
        if word.strip().lower() in segmented_text:
            likeliehood_score += 1

    return likeliehood_score


# ---- Data loading
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")

# ---- Inference
df["likeliehood_hospitalization"] = df["observationBlob"].apply(
    lambda x: get_likeliehood(x, hospitalization_keywords)
)

df["likeliehood_radiochemo"] = df["observationBlob"].apply(
    lambda x: get_likeliehood(x, radiotherapy_chemotherapy_keywords)
)

df["prediction"] = np.where(
    df["likeliehood_radiochemo"] > df["likeliehood_hospitalization"], 1, 0
)

# ---- Compute metrics
accuracy = accuracy_score(df[TARGET_LABEL], df["prediction"])
f1 = f1_score(df[TARGET_LABEL], df["prediction"])

print(f"Accuracy: {accuracy:.3f}")  # 0.439
print(f"F1 Score: {f1:.3f}")  # 0.400

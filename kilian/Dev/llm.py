from transformers import pipeline
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split 


# CONFIG
DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"

model_id = "openai/gpt-oss-20b"


# DATA LOADING
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# used for assay
df_test = df_test.head(10)
df_train = df_train.head(3)


# INFERENCE
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "I will give you a medical clinical note of a patient visit that has been summed up by a LLM, I need you to answer me \"1\" if the patient had a visit for a chemotherapy or a radiotherapy, or \"0\" if the patient were hospitalized. Here is the medical note : \n" + df_train[0, "txt_rw"]},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

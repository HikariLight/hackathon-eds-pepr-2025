import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal




# CONFIG
DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"

MODEL_ID = "openai/gpt-oss-20b"

chuck_size = 50
preds = []


# DATA LOADING
df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# assay
df_train = df_train.head(10)


# INFERENCE

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_ID)
)

# Simple classification
prediction = model(
    f"Using the following note medical note, can you tell me if the patient came for a chemo or radiotherapy or if the patient were hospitalized :\n {df_train["txt_rw"].iloc[0]}",
    Literal["Chemotherapy or Radiotherapy", "Hopitalization"]
)
print(prediction)












# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype="auto",
#     device_map="auto"
# )

# messages = [
#     {"role": "user", "content": f"I will give you a clinical note of a patient visit that was summarized by a LLM, I need you to answer me only the character \"1\" if the patient had a visit for a chemotherapy or a radiotherapy, or only \"0\" if the patient were hospitalized. Here is the medical note : \n {row['txt_rw']}"}
#  for _, row in df_train.iterrows()]


# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )

# for answer in outputs:
#     print(answer["generated_text"][-1])
#     preds.append(answer["generated_text"][-1].strip())

# df_train["prediction"] = preds



# # METRICS 
# true_results = df_train[TARGET_LABEL].apply(lambda x: str(x).strip())

# accuracy = accuracy_score(true_results, df["prediction"])
# f1 = f1_score(true_results, df["prediction"])

# print(f"Accuracy: {accuracy:.3f}")  # 0.439
# print(f"F1 Score: {f1:.3f}")  # 0.400





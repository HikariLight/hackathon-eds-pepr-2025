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
# DATA_PATH = "/home/kilian/Documents/programs/courriers_medics/kilian"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"

MODEL_ID = "openai/gpt-oss-20b"

NB_TRAIN = 100

preds = []


# DATA LOADING
df_train = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# assay
hospit_few = df_train[df_train[TARGET_LABEL] == 0].head(1)
chemo_few = df_train[df_train[TARGET_LABEL] == 1].head(1)

df_train = df_train.head(NB_TRAIN)


# INFERENCE

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_ID),
)

template = """---
Note: %n
Result: %r"""

few_shot_str = ""

for _, row in hospit_few.iterrows():
    few_shot_str += template.replace("%n", str(row["txt_rw"])).replace("%r", "Hopitalization")
for _, row in chemo_few.iterrows():
    few_shot_str += template.replace("%n", str(row["txt_rw"])).replace("%r", "Chemotherapy or Radiotherapy")

prompt = """
You are a medical assistant at a hospital. Your job is to verify whether a person came into the hospital for a hospitaization, or a radiotherapy/chemotherapy treatment. Read the example clinical note very carefully and decide whether the patient was hospitalized or administered treatment.

Here are some examples to help you:

EXAMPLES

Here is the clinical note to classify:

CLINICAL NOTE

"""




for _, row in df_train.iterrows():
    prediction = model(
        prompt.replace("EXAMPLES", few_shot_str).replace("CLINICAL NOTE", row["txt_rw"]),
        Literal["Chemotherapy or Radiotherapy", "Hopitalization"],
    )
    preds.append(prediction)

df_train["prediction"] = [
    1 if x == "Chemotherapy or Radiotherapy" else 0 for x in preds
]


# METRICS
true_results = df_train[TARGET_LABEL].apply(lambda x: int(x))

accuracy = accuracy_score(true_results, df_train["prediction"])
f1 = f1_score(true_results, df_train["prediction"])

print(f"Accuracy: {accuracy:.3f}")  # 0.439
print(f"F1 Score: {f1:.3f}")  # 0.400


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

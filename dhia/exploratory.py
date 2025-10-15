import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

for key, value in os.environ.items():
    print(f"{key}: {value}")

api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")
print(api_key)

# # CONFIG
# DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
# data_file = "hackathon_train.csv"
# TARGET_LABEL = "seance_chimio"

# df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")

# print(df.columns)

# print(df["class_group"].value_counts())

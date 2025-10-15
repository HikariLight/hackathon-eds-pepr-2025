import pandas as pd
import os

# CONFIG
DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"
TARGET_LABEL = "seance_chimio"

df = pd.read_csv(os.path.join(DATA_PATH, data_file), sep=";")

print(df.columns)

print(df["class_group"].value_counts())

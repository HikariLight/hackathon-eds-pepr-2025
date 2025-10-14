import pandas
import os

# CONFIG

DATA_PATH = "/mnt/eds_projets/inria_hackathon/data"
data_file = "hackathon_train.csv"

df = pandas.read_csv(os.path.join(DATA_PATH, data_file))
print(df.columns)

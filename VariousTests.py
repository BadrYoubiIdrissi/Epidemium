# %%
import pandas as pd

# %%

df = pd.read_csv("BDD/Test/biostats.csv")
print(df.groupby(["Height"]).mean())

# %%

from DataPreparation.Database import Database

db = Database("BDD/Test/biostats.csv")
chunks = db.sliceToChunks('Age')
print(chunks)
# %% Imports

from DataPreparation.Database import Database
from LSTM.NN import NN

# %% Loading database

RawDB = Database("BDD/Prepared/AllFeaturesRenamed.csv")

# %% Normalising Mortality

RawDB.Mortality_sum = RawDB.Mortality_sum*1e6 / RawDB["Population, total"]

chunks = RawDB.sliceToChunks("year")
from DataPreparation.Database import Database
import tensorflow as tf

db = Database("BDD/Test/biostats.csv")

test = db.sliceToChunks('Sex')

for key in test:
    print(test[key])

# Imports

from DataPreparation.Database import Database
from LSTM.NN import LstmNN
import numpy as np
import matplotlib.pyplot as plt

# Loading database

RawDB = Database("BDD/Prepared/AllFeaturesNormalizedFilled.csv")

# Normalising Mortality

chunks = RawDB.sliceToChunks("area")

for area in chunks:
    chunks[area] = chunks[area].drop(["year"], axis=1).values

# LSTM model

windowSize = 5
nbFeatures = chunks["Australia"].shape[1]
neurons = 256, 256

nn = LstmNN(windowSize, nbFeatures, neurons)

for area in chunks:
    if len(chunks[area]) > 2*windowSize:
        train_raw, test_raw = nn.splitTrainTest(chunks[area], 0.8)
        train_scaled = nn.scale(train_raw)
        test_scaled = nn.scaler.transform(test_raw)
        train = nn.toSupervised(train_scaled)
        test = nn.toSupervised(test_scaled)
    
        train_in , train_out = nn.inputOutput(train)
        test_in , test_out = nn.inputOutput(test)
    
        nn.model.fit(x=train_in,y=train_out,epochs=100, shuffle=False)

pred = nn.prediction(test_in[0,:,:], 100)

plt.plot(pred[:,-1])
plt.plot(test_out[:,-1])
plt.show()
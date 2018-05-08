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
neurons = 256, 512

nn = LstmNN(windowSize, nbFeatures, neurons)
train_in, train_out = {}, {}
test_in, test_out = {}, {}

for area in chunks:
    if len(chunks[area]) > 2*windowSize:
        train_raw, test_raw = nn.splitTrainTest(chunks[area], 0.8)
        train_scaled = nn.scale(train_raw)
        test_scaled = nn.scaler.transform(test_raw)
        train = nn.toSupervised(train_scaled)
        test = nn.toSupervised(test_scaled)
    
        train_in[area] , train_out[area] = nn.inputOutput(train)
        test_in[area] , test_out[area] = nn.inputOutput(test)
    
        nn.model.fit(x=train_in[area],y=train_out[area],epochs=1000, shuffle=False)

def predictForArea(area):
    pred = nn.scaler.inverse_transform(nn.model.predict(test_in[area]))
    predtr = nn.scaler.inverse_transform(nn.model.predict(train_in[area]))
    plt.plot(np.concatenate((predtr[:,-1], pred[:,-1]), axis=0))
    plt.plot(np.concatenate((nn.scaler.inverse_transform(train_out[area])[:,-1], nn.scaler.inverse_transform(test_out[area])[:,-1]), axis=0))
    plt.show()
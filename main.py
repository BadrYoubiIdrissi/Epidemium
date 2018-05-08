# %% Imports

from DataPreparation.Database import Database
from LSTM.NN import LstmNN
import numpy as np
import matplotlib.pyplot as plt

# %% Loading database

RawDB = Database("BDD/Prepared/AllFeaturesRenamed.csv")

# %% Normalising Mortality

RawDB.Mortality_sum = RawDB.Mortality_sum*1e6 / RawDB["Population, total"]

chunks = RawDB.sliceToChunks("year")

# %% LSTM model

windowSize = 40
nbFeatures = 2
neurons = 100, 256

nn = LstmNN(windowSize, nbFeatures, neurons)

x = np.linspace(0,20*np.pi,1000)
y = np.concatenate((nn.to2D(100*np.sin(x)),nn.to2D(100*np.cos(x))), axis=1)

train_raw, test_raw = nn.splitTrainTest(y, 0.8)
train_scaled = nn.scale(train_raw)
test_scaled = nn.scaler.transform(test_raw)
train = nn.toSupervised(train_scaled)
test = nn.toSupervised(test_scaled)

train_in , train_out = nn.inputOutput(train)
test_in , test_out = nn.inputOutput(test)

nn.model.fit(x=train_in,y=train_out,epochs=100, shuffle=False, batch_size=50)

pred = nn.prediction(test_in[0,:,:], 100)

plt.plot(pred[:,0])
plt.plot(pred[:,1])
plt.show()
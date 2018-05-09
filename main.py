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

nbFeatures = chunks["Australia"].shape[1]

train_in, train_out = {}, {}
test_in, test_out = {}, {}


def predictForArea(area, windowSize, fileName=None, neurons = (50, 50)):
    
    nn = LstmNN(windowSize, nbFeatures, neurons)
    data = chunks[area]
    prop = 0.8
    if len(data) > 2*windowSize:
        train_raw, _ = nn.splitTrainTest(data, prop)
        nn.fitScaler(train_raw)
        data = nn.scaler.transform(data)
        supData = nn.toSupervised(data)
        train, test = nn.splitTrainTest(supData, prop)
    
        train_in[area] , train_out[area] = nn.inputOutput(train)
        test_in[area] , test_out[area] = nn.inputOutput(test)
    
        nn.model.fit(x=train_in[area],y=train_out[area],epochs=100, shuffle=False, verbose=0)
    
        predtr = nn.scaler.inverse_transform(nn.model.predict(train_in[area]))
        pred = nn.prediction(test_in[area][0,:,:], len(chunks[area])-(windowSize+len(predtr)))
        plt.plot(range(windowSize,windowSize+len(predtr)), predtr[:,-1], label="Predicted on training set")
        plt.plot([windowSize+len(predtr)-1, windowSize+len(predtr)], [predtr[-1,-1], pred[0,-1]], "b")
        plt.plot(range(windowSize+len(predtr),len(chunks[area])), pred[:,-1], label="Predicted on test set")
        plt.plot(chunks[area][:,-1], label="Real")
        plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()
    else:
        print("Too little data")

for i in range(3,10):
    predictForArea("Mexico", i, "Figures/Percountry/100epochs/window{}.png".format(i))
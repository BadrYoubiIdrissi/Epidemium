# Imports

from DataPreparation.Database import Database
from LSTM.NN import LstmNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DataPreparation.Utils import *
from keras.utils import plot_model
from DynamicUpdate import DynamicUpdate

#== Parameters

fileName = "Figures/Viable/Results/{}.png"
neurons = (30, 10)
lag = 2
windowSize = 4
feature = -1
yearOffset = 0

# Loading database

db = Database("BDD/Prepared/AllFeaturesNormalizedFilled.csv", windowSize, lag)

db.sliceToChunks("area")

db.dropCol("year")

db.toNumpyArr()

# LSTM model

nbFeatures = db.nbFeatures

nn = {}
results = {}

areas = db.getChunksKeys()

testAr = "Mexico"
    
data = db.chunks[testAr][yearOffset:]
viable = list(db.chunks.keys())
viable.remove(testAr)
nn[testAr] = LstmNN(windowSize, nbFeatures, neurons)

db.buildTrainTestSets(1)

db.buildCumulatedTrainTestSets(viable)

d = DynamicUpdate(len(data)+lag)

d.ax.plot(data[:,feature], label="Real")
d.ax.plot(range(windowSize+lag,windowSize+lag+len(db.train_out[testAr][:,-1])), 
          db.resultScaler[testAr].inverse_transform(to2D(db.train_out[testAr][:,-1])))

for i in range(30):
    nn[testAr].model.fit(x=db.cum_train_in,y=db.cum_train_out,epochs = 1, batch_size = 100, verbose=1)
    da = db.toSupervisedTest(db.fullScaler[testAr].transform(data))
    predtr = db.resultScaler[testAr].inverse_transform(to2D(nn[testAr].model.predict(da)))
    x = range(windowSize+lag,windowSize+lag+len(predtr))
    d.on_running(x, predtr[:,-1])

#nn[testAr].model.save("Models/{}".format(testAr))
    

# if fileName:
#     plt.savefig(fileName.format(testAr))
# print(testAr)
        
    
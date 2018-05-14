# Imports

from DataPreparation.Database import Database
from LSTM.NN import LstmNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DataPreparation.Utils import *
from keras.utils import plot_model

#== Parameters

fileName = "Figures/Viable/Results/{}.png"
neurons = (50, 100)
windowSize = 4
feature = -1
yearOffset = 0

# Loading database

db = Database("BDD/Prepared/AllFeaturesNormalizedFilled.csv", windowSize)

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
viable = areas[:]
viable.remove(testAr)
nn[testAr] = LstmNN(windowSize, nbFeatures, neurons)

db.buildTrainTestSets(1)

db.buildCumulatedTrainTestSets(viable)

nn[testAr].model.fit(x=db.cum_train_in,y=db.cum_train_out,epochs = 100, verbose=1)
nn[testAr].model.save("Models/{}.h5".format(testAr))

predtr = db.resultScaler[testAr].inverse_transform(nn[testAr].model.predict(db.train_in[testAr]))

plt.plot(data[:,feature], label="Real")
plt.plot(range(windowSize,windowSize+len(predtr)), predtr[:,feature], label="Predicted")

plt.legend()
plt.show()
# if fileName:
#     plt.savefig(fileName.format(testAr))
# print(testAr)
        
    
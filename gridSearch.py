# Imports

from DataPreparation.Database import Database
from LSTM.NN import LstmNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DataPreparation.Utils import *
from keras.utils import plot_model

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

#== Parameters

#fileName="Figures/Viable/200epochs/{}.png"
#fileName = "Figures/Viable/Results/{}.png"
fileName = None
feature = -1
yearOffset = 0
nn = {}
results = {}
for ni in [20,50,100,250]:
    for nj in [20,50,100,250]:
        for w in [2,4,6,10]:
            neurons = (ni, nj)
            windowSize = w

            #areas = ["Argentina", "Brazil", "Ecuador", "El Salvador", "Chile", "Colombia", "Costa Rica", "Guyana", "Jamaica", "Mexico", "Guatemala", "Cuba", "Honduras", "Peru", "Nicaragua", "Panama", "Paraguay", "Uruguay", "Dominican Republic", "Haiti", "Puerto Rico"]
            areas = list(chunks.keys())
            #viable = ["Australia","Austria","Belgium","Canada","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Iceland","Ireland","Israel","Italy","Japan","Netherlands","New Zealand","Norway","Portugal","Slovakia","Slovenia","South Korea","Spain","Sweden","Singapore","Switzerland","United Kingdom","United States"]
            testAr = "Mexico"
            data = chunks[testAr][yearOffset:]
            if len(data) > windowSize:
                viable = areas[:]
                viable.remove(testAr)
                nn[testAr] = LstmNN(windowSize, nbFeatures, neurons)
                cum_train_in = np.zeros((0,windowSize, nbFeatures))
                cum_train_out = np.zeros((0,1))
                scaler = {}

                for area in areas:
                    data = chunks[area][yearOffset:]
                    prop = 1
                    if len(data) > windowSize:
                        scalerTot = MinMaxScaler(feature_range=(-1,1))
                        scaler[area] = MinMaxScaler(feature_range=(-1,1))
                        t = splitTrainTest(data, prop)[0]
                        scalerTot.fit(t)
                        scaler[area].fit(to2D(t[:,feature]))
                        data = scalerTot.transform(data)
                        
                        supData = nn[testAr].toSupervised(data)
                        train, test = splitTrainTest(supData, prop)
                    
                        train_in[area] , train_out[area] = inputOutput(train, feature)
                        test_in[area] , test_out[area] = inputOutput(test, feature)
                    
                        if area in viable:
                            cum_train_in = np.concatenate((cum_train_in, train_in[area]), axis=0)
                            cum_train_out = np.concatenate((cum_train_out, train_out[area]), axis=0)

                nn[testAr].model.fit(x=cum_train_in,y=cum_train_out,epochs = 100, verbose=1)
                #nn[testAr].model.save("Models/{}.h5".format(testAr))

                data = chunks[testAr][yearOffset:]

                predtr = scaler[testAr].inverse_transform(nn[testAr].model.predict(train_in[testAr]))
                #pred = nn[testAr].prediction(train_in[testAr][-1,:,:], len(chunks[testAr])-(windowSize+len(predtr)))
                #pred = scaler[testAr].inverse_transform(nn[testAr].model.predict(test_in[testAr]))
                #predRoll = scaler[testAr].inverse_transform(nn[testAr].rollingWindowPrediction(train_in[testAr][-1,:,:], 10))
                
                plt.plot(range(windowSize,windowSize+len(predtr)), predtr[:,feature], label="Predicted")
                
                #plt.plot([windowSize+len(predtr)-1, windowSize+len(predtr)], [predtr[-1,feature], pred[0,feature]], "orange")
                #plt.plot([windowSize+len(predtr)-1, windowSize+len(predtr)], [predtr[-1,feature], predRoll[0,feature]], "green")
                
                #plt.plot(range(windowSize+len(predtr),len(data)), pred[:,feature], label="Predicted on test set")
                #plt.plot(range(windowSize+len(predtr),windowSize+len(predtr)+len(predRoll)), predRoll[:,feature], label="Predicted on rolling window")

                plt.plot(data[:,feature], label="Real")
                plt.legend()
                if fileName:
                    plt.savefig(fileName.format(testAr))
                print(testAr)
                plt.show()

                print(testAr)
                results[ni,nj,w] = nn[testAr].model.evaluate(train_in[testAr], train_out[testAr],verbose=1)
                print(results[ni,nj,w])
        

#for i in range(3,10):
    #predictForArea("Mexico", i, "Figures/Percountry/100epochs/window{}.png".format(i))
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


def predictForArea(area, windowSize, feature = -1, fileName=None, neurons = (256, 100)):
    
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
    
        nn.model.fit(x=train_in[area],y=train_out[area],epochs=100)
            
        predtr = nn.scaler.inverse_transform(nn.model.predict(train_in[area]))
        #pred = nn.prediction(train_in[area][-1,:,:], len(chunks[area])-(windowSize+len(predtr)))
        pred = nn.scaler.inverse_transform(nn.model.predict(test_in[area]))
        predRoll = nn.prediction(train_in[area][-1,:,:], 10)
        
        plt.plot(range(windowSize,windowSize+len(predtr)), predtr[:,feature], label="Predicted on training set")
        
        plt.plot([windowSize+len(predtr)-1, windowSize+len(predtr)], [predtr[-1,feature], pred[0,feature]], "b")
        
        plt.plot(range(windowSize+len(predtr),len(chunks[area])), pred[:,feature], label="Predicted on test set")
        plt.plot(range(windowSize+len(predtr),windowSize+len(predtr)+len(predRoll)), predRoll[:,feature], label="Predicted on rolling window")
      
        plt.plot(chunks[area][:,feature], label="Real")
        plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()
    else:
        print("Too little data")

fileName=None
neurons = (256, 100)
windowSize = 10
feature = -1
areas = ["Argentina", "Brazil", "Ecuador", "El Salvador", "Chile", "Colombia", "Costa Rica", "Guyana", "Jamaica", "Mexico", "Guatemala", "Cuba", "Honduras", "Peru", "Nicaragua", "Panama", "Paraguay", "Uruguay", "Dominican Republic", "Haiti", "Puerto Rico"]
nn = LstmNN(windowSize, nbFeatures, neurons)
cum_train_in = np.zeros((0,windowSize, nbFeatures))
cum_train_out = np.zeros((0,nbFeatures))
train_raw = np.zeros((0,nbFeatures))
for area in areas:
    data = chunks[area]
    prop = 0.8
    if len(data) > windowSize:
        supData = nn.toSupervised(data)
        train, test = nn.splitTrainTest(supData, prop)
    
        train_in[area] , train_out[area] = nn.inputOutput(train)
        test_in[area] , test_out[area] = nn.inputOutput(test)
        
        cum_train_in = np.concatenate((cum_train_in, train_in[area]), axis=0)
        cum_train_out = np.concatenate((cum_train_out, train_out[area]), axis=0)
        train_raw = np.concatenate((train_raw, nn.splitTrainTest(data, prop)[0]), axis=0)
        
nn.fitScaler(train_raw)
cum_train_out = nn.scaler.transform(cum_train_out)
for i in range(cum_train_in.shape[1]):
    cum_train_in[:,i,:] = nn.scaler.transform(cum_train_in[:,i,:])
nn.model.fit(x=cum_train_in,y=cum_train_out,epochs=500, shuffle=True)

for area in areas:
    data = chunks[area]
    if len(data) > windowSize:
        predtr = nn.scaler.inverse_transform(nn.model.predict(train_in[area]))
        #pred = nn.prediction(train_in[area][-1,:,:], len(chunks[area])-(windowSize+len(predtr)))
        pred = nn.scaler.inverse_transform(nn.model.predict(test_in[area]))
        #predRoll = nn.prediction(train_in[area][-1,:,:], 10)
        
        plt.plot(range(windowSize,windowSize+len(predtr)), predtr[:,feature], label="Predicted on training set")
        
        plt.plot([windowSize+len(predtr)-1, windowSize+len(predtr)], [predtr[-1,feature], pred[0,feature]], "b")
        
        plt.plot(range(windowSize+len(predtr),len(chunks[area])), pred[:,feature], label="Predicted on test set")
        #plt.plot(range(windowSize+len(predtr),windowSize+len(predtr)+len(predRoll)), predRoll[:,feature], label="Predicted on rolling window")
      
        plt.plot(chunks[area][:,feature], label="Real")
        plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()



#for i in range(3,10):
    #predictForArea("Mexico", i, "Figures/Percountry/100epochs/window{}.png".format(i))
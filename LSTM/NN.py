from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout, Activation, TimeDistributed
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LstmNN():
    def __init__(self, windowSize, nbFeatures, neurons = (50, 256), dropout=0.5):
        self.windowSize = windowSize
        self.neurons = neurons
        self.nbFeatures = nbFeatures

        self.model = Sequential()
        self.model.add(CuDNNLSTM(neurons[0], input_shape = (windowSize, nbFeatures), return_sequences=True))
        self.model.add(Dropout(dropout))
        self.model.add(CuDNNLSTM(neurons[1]))
        self.model.add(Dense(nbFeatures)) 
        self.model.add(Activation("linear"))
        self.model.compile(loss="mse", optimizer="adam")
        self.model.summary()
        
    def toSupervised(self, a):
        orig = a.copy() #On garde une copie car on va modifier a
        a = a[self.windowSize:] #On enlève les occurences qui n'ont pas assez de valeurs dans le passé
        sh = a.shape
        a = a.reshape((sh[0],1,sh[1])) #On reshape a pour être dans la forme exigée par Keras
        for i in range(self.windowSize):
            b = orig.copy()
            b = np.roll(b, 1+i, axis=0) #On décale b de 1 unité en temps
            b = b[self.windowSize:] #On donne à b la même forme que a
            b = b.reshape((sh[0],1,sh[1]))
            a = np.concatenate((b,a), axis=1) #On rajoute sur la deuxième dimmension les décalages temporels
        return a
    
    def rollingWindowPrediction(self, startingWindow, nPred):
        rollingWindow = startingWindow
        pred = np.zeros((nPred, self.nbFeatures))
        for i in range(nPred):
            p = self.model.predict(self.threeDimInput(rollingWindow))
            pred[i] = p
            rollingWindow = np.concatenate((rollingWindow, p), axis=0)[1:]
        return self.to2D(np.array(pred))

    def to2D(self, a):
        if a.ndim == 1: 
            return a.reshape((a.shape[0],1))
        else:
            return a

    def fitScaler(self, a):
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.scaler.fit(a)

    def inputOutput(self, a):
        return a[:,:-1,:], a[:,-1,:]

    def threeDimInput(self, a):
        if a.ndim == 2:
            return a.reshape((1, a.shape[0], a.shape[1]))
        else:
            return a.reshape((1, a.shape[0], 1))

    def splitTrainTest(self, y, propTrain):
        self.propTrain = propTrain
        m = int(propTrain*len(y))
        return self.to2D(y[:m]), self.to2D(y[m:])

    def prediction(self, startingWindow,nbPred):
        return self.scaler.inverse_transform(self.rollingWindowPrediction(startingWindow, nbPred))

"""

Author : Badr YOUBI IDRISSI

This file will contain the Database object which loads the different
dataframes needed and prepares them
Different functionalities include : 
    - load from csv
    - sliceToChunks(column) : slices the dataframe to chunks with the same value for "column"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DataPreparation.Utils import *

class Database():
    def __init__(self, filepath, windowSize, lag=0, mainFeature=-1):
        self.df = pd.read_csv(filepath)
        self.windowSize = windowSize
        self.lag = lag
        self.mainFeature = mainFeature
        self.nbFeatures = self.df.shape[1]
        self.train_in = {}
        self.train_out = {}
        self.test_in = {}
        self.test_out = {}
        self.fullScaler = {}
        self.resultScaler = {}

    def sliceToChunks(self, column, dropCol = True):
        chunks = {}
        values = self.df[column].unique()
        if dropCol:
            self.nbFeatures -= 1
            for v in values:
                chunk = self.df[self.df[column] == v].drop(column, axis=1)
                chunks[v] = chunk
        else:
            for v in values:
                chunk = self.df[self.df[column] == v]
                chunks[v] = chunk
        self.chunks = chunks
        
    def dropCol(self, col):
        self.nbFeatures -= 1
        for area in self.chunks:
            self.chunks[area] = self.chunks[area].drop([col], axis=1)
        
    def toNumpyArr(self):
        for area in self.chunks:
            self.chunks[area] = self.chunks[area].values
    
    def getChunksKeys(self):
        return list(self.chunks.keys())
    
    def getAreasIfSizeMin(self, minSize, a = None):
        if not a:
            a = self.chunks.keys()
        areas = []
        for area in a:
            if len(self.chunks[area]) > minSize:
                areas.append(area)
        return areas

    def toSupervised(self, a):
        orig = a.copy() #On garde une copie car on va modifier a
        c = a[self.windowSize + self.lag:] #On enlève les occurences qui n'ont pas assez de valeurs dans le passé
        sh = c.shape
        c = c.reshape((sh[0],1,sh[1])) #On reshape a pour être dans la forme exigée par Keras
        for i in range(self.windowSize + self.lag):
            b = orig.copy()
            b = np.roll(b, 1+i, axis=0) #On décale b de 1 unité en temps
            b = b[self.windowSize + self.lag:] #On donne à b la même forme que a
            b = b.reshape((sh[0],1,sh[1]))
            c = np.concatenate((b,c), axis=1) #On rajoute sur la deuxième dimmension les décalages temporels
        return c
    
    def toSupervisedTest(self, a):
        orig = a.copy() #On garde une copie car on va modifier a
        c = a[self.windowSize-1:]
        sh = c.shape
        c = np.zeros((sh[0],0,sh[1])) #On reshape a pour être dans la forme exigée par Keras
        for i in range(self.windowSize):
            b = orig.copy()
            b = np.roll(b, i, axis=0) #On décale b de 1 unité en temps
            b = b[self.windowSize-1:] #On donne à b la même forme que a
            b = b.reshape((sh[0],1,sh[1]))
            c = np.concatenate((b,c), axis=1) #On rajoute sur la deuxième dimmension les décalages temporels
        return c
    
    def inputOutput(self, a):
        return a[:,:self.windowSize,:], a[:,-self.windowSize:,self.mainFeature]
    
    def toOrigScale(self, area, a):
        return self.resultScaler[area].inverse_transform(a[area])
    
    def buildTrainTestSets(self, prop):
        areas = self.getAreasIfSizeMin(self.windowSize+self.lag)
        for area in areas:
            data = self.chunks[area]

            self.fullScaler[area] = MinMaxScaler(feature_range=(-1,1))
            self.resultScaler[area] = MinMaxScaler(feature_range=(-1,1))
            trainingData = splitTrainTest(data, prop)[0]
            self.fullScaler[area].fit(trainingData) #We fit only on training data
            self.resultScaler[area].fit(to2D(trainingData[:,self.mainFeature]))
            data = self.fullScaler[area].transform(data) #We transform the whole of data
            
            supData = self.toSupervised(data)
            train, test = splitTrainTest(supData, prop)
        
            self.train_in[area] , self.train_out[area] = self.inputOutput(train)
            self.test_in[area] , self.test_out[area] = self.inputOutput(test)
            
    
    def buildCumulatedTrainTestSets(self, viable):
        self.cum_train_in = np.zeros((0,self.windowSize, self.nbFeatures))
        self.cum_train_out = np.zeros((0,self.windowSize,1))
        for area in self.getAreasIfSizeMin(self.windowSize+self.lag, viable):
            self.cum_train_in = np.concatenate((self.cum_train_in, to3D(self.train_in[area])), axis=0)
            self.cum_train_out = np.concatenate((self.cum_train_out, to3D(self.train_out[area])), axis=0)
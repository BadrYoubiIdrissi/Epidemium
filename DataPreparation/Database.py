"""

Author : Badr YOUBI IDRISSI

This file will contain the Database object which loads the different dataframes needed and prepares them

Different functionalities include : 
    - load from csv
    - sliceToChunks(column) : slices the dataframe to chunks with the same value for "column"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DataPreparation.Utils import *

class Database():
    def __init__(self, filepath, windowSize, mainFeature = -1):
        self.df = pd.read_csv(filepath)
        self.windowSize = windowSize
        self.mainFeature = mainFeature
        self.nbFeatures = self.df.shape[1]
        self.train_in = {}
        self.train_out = {}
        self.test_in = {}
        self.test_out = {}
        self.scaler = {}

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

    def buildTrainTestSets(self, prop):
        areas = self.getAreasIfSizeMin(self.windowSize)
        for area in areas:
            data = self.chunks[area]

            self.scaler[area] = MinMaxScaler(feature_range=(-1,1))
            self.scaler[area].fit(splitTrainTest(data, prop)[0]) #We fit only on training data
            
            data = self.scaler[area].transform(data) #We transform the whole of data
            
            supData = self.toSupervised(data)
            train, test = splitTrainTest(supData, prop)
        
            self.train_in[area] , self.train_out[area] = inputOutput(train, self.mainFeature)
            self.test_in[area] , self.test_out[area] = inputOutput(test, self.mainFeature)
            
    
    def buildCumulatedTrainTestSets(self, viable):
        self.cum_train_in = np.zeros((0,self.windowSize, self.nbFeatures))
        self.cum_train_out = np.zeros((0,1))
        for area in self.getAreasIfSizeMin(self.windowSize, viable):
            self.cum_train_in = np.concatenate((self.cum_train_in, self.train_in[area]), axis=0)
            self.cum_train_out = np.concatenate((self.cum_train_out, self.train_out[area]), axis=0)
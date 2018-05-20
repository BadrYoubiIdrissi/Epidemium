# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:45:28 2018

@author: Badr Youbi Idrissi
"""

def to2D(a):
    if a.ndim == 1: 
        return a.reshape((a.shape[0],1))
    elif a.ndim == 3:
        return a.reshape((a.shape[0],a.shape[1]))
    else:
        return a

def to3D(a):
    if a.ndim == 2: 
        return a.reshape((a.shape[0],a.shape[1],1))
    elif a.ndim == 1:
        return a.reshape((a.shape[0],1,1))
    else:
        return a

def threeDimInput(a):
    if a.ndim == 2:
        return a.reshape((1, a.shape[0], a.shape[1]))
    else:
        return a.reshape((1, a.shape[0], 1))

def splitTrainTest(y, propTrain):
    propTrain = propTrain
    m = int(propTrain*len(y))
    return y[:m], y[m:]
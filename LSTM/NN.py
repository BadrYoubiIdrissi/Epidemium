from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout, Activation
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

y = np.sin(np.linspace(0,20*np.pi,1000))

def to2D(a):
    if a.ndim == 1: 
        return a.reshape((a.shape[0],1))
    else:
        return a

def scale(a):
    """"""
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled = scaler.fit_transform(a) #On normalise
    return scaler, scaled
def toSupervised(a, windowSize):
    orig = a.copy() #On garde une copie car on va modifier a
    a = a[windowSize:] #On enlève les occurences qui n'ont pas assez de valeurs dans le passé
    sh = a.shape
    a = a.reshape((sh[0],1,sh[1])) #On reshape a pour être dans la forme exigée par Keras
    for i in range(windowSize):
        b = orig.copy()
        b = np.roll(b, 1+i, axis=0) #On décale b de 1 unité en temps
        b = b[windowSize:] #On donne à b la même forme que a
        b = b.reshape((sh[0],1,sh[1]))
        a = np.concatenate((b,a), axis=1) #On rajoute sur la deuxième dimmension les décalages temporels
    return a

def inputOutput(a):
    return a[:,:-1,:], a[:,-1,:]

def threeDimInput(a):
    if a.ndim == 2:
        return a.reshape((1, a.shape[0], a.shape[1]))
    else:
        return a.reshape((1, a.shape[0], 1))

def rollingWindowPrediction(startingWindow, feature=0):
    rollingWindow = startingWindow
    nPred = 1000
    pred = []
    for i in range(nPred):
        p = model.predict(threeDimInput(rollingWindow))
        pred.append(p[0,feature])
        rollingWindow = np.concatenate((rollingWindow, p), axis=0)[1:]
    return to2D(np.array(pred))


windowSize = 40
nbFeatures = 1
neurons = 50

model = Sequential()
model.add(CuDNNLSTM(neurons, input_shape = (windowSize, nbFeatures), return_sequences=True))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(256))
model.add(Dense(nbFeatures))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")
model.summary()

train_raw, test_raw = to2D(y[:800]), to2D(y[800:])
scaler, train_scaled = scale(train_raw)
test_scaled = scaler.transform(test_raw)
train = toSupervised(train_scaled, windowSize)
test = toSupervised(test_scaled, windowSize)

train_in , train_out = inputOutput(train)
test_in , test_out = inputOutput(test)

model.fit(x=train_in,y=train_out,epochs=100, shuffle=False)

pred = scaler.inverse_transform(rollingWindowPrediction(test_in[0,:,:]))
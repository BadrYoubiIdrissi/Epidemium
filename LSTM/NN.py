from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout, Activation
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

y = np.linspace(0,10*np.pi,1000)**2
print(y.shape)
# plt.plot(y, label="Training data")

def toSupervised(a, windowSize):
    scaler = MinMaxScaler(feature_range=(-1,1))
    if a.ndim == 1: #MinMaxScaler n'accepte que les tableaux de dimmension 2
        a = a.reshape(a.shape[0],1)
    a = scaler.fit_transform(a) #On normalise
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
    return a, scaler

def inputOutput(a):
    return a[:,:-1,:], a[:,-1,:]

windowSize = 40
nbFeatures = 1
neurons = 50
data, scaler = toSupervised(y, windowSize)
train, test = data[:800], data[800:]

# for i in range(train.shape[1]):
#     plt.plot(test[:,i,:])
# plt.show()

model = Sequential()
model.add(CuDNNLSTM(neurons, input_shape = (windowSize, nbFeatures), return_sequences=True))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(256))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")
model.summary()

train_in , train_out = inputOutput(train)
test_in , test_out = inputOutput(test)

model.fit(x=train_in,y=train_out,epochs=100, shuffle=False)

plt.plot(model.predict(train_in))
plt.plot(train_out)
plt.show()

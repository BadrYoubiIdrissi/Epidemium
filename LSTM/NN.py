from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.linspace(0,1,1000)
Y = X**2
N = len(X)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

plt.plot(X_test, Y_test,".")


hidden_units = 1
input_size = 1

model = Sequential()
model.add(Dense(4, input_dim=1, kernel_initializer='normal', activation='tanh'))
model.add(Dense(6, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(optimizer = 'adam', loss = 'mse')


model.fit(x=X_train, y=Y_train, epochs=50, batch_size=5, callbacks=[tbCallBack])
print(model.evaluate(X_test, Y_test))
X = np.linspace(0,2,1000)
plt.plot(X, model.predict(X))
plt.show()
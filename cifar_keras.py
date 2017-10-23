import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras import optimizers
import numpy as np
np.random.seed(777)  # for reproducibility
import os
import pickle
import numpy as np

from keras.datasets import cifar10
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

sgd = optimizers.SGD(lr=0.1)
model.add(Dense(200, input_dim=3072, activation = 'sigmoid'))
model.add(Dense(10, input_dim=200, activation = 'sigmoid'))
model.summary()
model.compile(loss='mean_squared_logarithmic_error',
             optimizer='sgd',
            metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size = 64, epochs=1)
score = model.evaluate(x_test, y_test)
print('\nAccuracy:', score[1])

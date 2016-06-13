from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys

from keras.layers import Dense
from keras.models import Sequential

from main import unpickle

batch_size = 2
input_dim = 3
midway_dim = 4
output_dim = 5
train_y_dim = 6
X_batch = np.arange(
    batch_size * input_dim).reshape(
    batch_size, input_dim)
Y_batch = np.arange(
    batch_size * midway_dim).reshape(
    batch_size, midway_dim)

encoder = Sequential()
encoder_layer = Dense(midway_dim, input_dim=input_dim)
encoder.add(encoder_layer)

encoder.compile(loss='categorical_crossentropy',
                optimizer='sgd')
encoder.train_on_batch(X_batch, Y_batch)

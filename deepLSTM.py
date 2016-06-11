from __future__ import print_function

import json
import os
import pickle
from functools import partial
import numpy as np
import theano
import theano.tensor as T
import keras
from keras.engine import Input
from keras.layers import Dense, GRU, SimpleRNN
from keras.models import Sequential

batch_size = 2
input_dim = 3
output_dim = 4
timesteps = 5
X_batch = np.arange(
    batch_size * timesteps * input_dim).reshape(
    batch_size, timesteps, input_dim)
Y_batch = np.arange(
    batch_size * timesteps * output_dim).reshape(
    batch_size, timesteps, output_dim)

model = Sequential()
model.add(SimpleRNN(output_dim,
                    input_shape=(timesteps, input_dim),
                    return_sequences=True))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.train_on_batch(X_batch, Y_batch)
# loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

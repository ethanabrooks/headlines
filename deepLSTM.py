from __future__ import print_function

import json
import os
import pickle
from functools import partial
import numpy as np
import theano
import theano.tensor as T
import keras
from keras.layers import Dense, GRU, SimpleRNN
from keras.models import Sequential
import seq2seq.layers.decoders
from seq2seq.models import Seq2seq
import cPickle

batch_size = 2
input_dim = 3
output_dim = 4
timesteps_i = 5
timesteps_o = 6
X_batch = np.arange(
    batch_size * timesteps_i * input_dim).reshape(
    batch_size, timesteps_i, input_dim)
Y_batch = np.arange(
    batch_size * timesteps_o * output_dim).reshape(
    batch_size, timesteps_o, output_dim)

model = Seq2seq(batch_input_shape=(batch_size, timesteps_i, input_dim),
                hidden_dim=7,
                output_length=timesteps_o,
                output_dim=output_dim,
                depth=2,
                peek=True)
# model.add(SimpleRNN(output_dim,
#                     input_shape=(timesteps, input_dim),
#                     return_sequences=True,
#                     unroll=True))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd')
model.train_on_batch(X_batch, Y_batch)
# loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

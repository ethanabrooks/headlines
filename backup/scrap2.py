from __future__ import print_function

import os
import pickle
from functools import partial

import numpy as np
import theano
from lasagne.layers import GRULayer, DenseLayer, InputLayer, get_output, EmbeddingLayer
from theano import tensor as T

bsize = 1
out_dim = 2
dim = 3
seq_len = 4

X = T.imatrix('X')
h = theano.shared(np.ones((1, out_dim)), 'h', allow_downcast=True)

input = InputLayer(shape=(bsize, seq_len),
                   input_var=X)
embed = EmbeddingLayer(input, bsize * seq_len, dim)
gru = GRULayer(incoming=embed, num_units=out_dim, hid_init=h)
output = get_output(gru)

x = np.arange(bsize * seq_len, dtype='int32').reshape(bsize, seq_len)
f = theano.function([X], output,
                    on_unused_input='ignore')

result = f(x)
if type(result) == list:
    for elt in result:
        print(result)
        print('-------------------')
else:
    print(result)

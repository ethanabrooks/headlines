from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys

instances = 2
nclasses = 3


def g(i):
    return T.tile(T.shape_padright(T.arange(i, i + nclasses)), instances).T


def f(i):
    return [i + 1, g(i)]


i0 = T.constant(0, dtype="int64")
[i_f, t], _ = theano.scan(fn=f,
                          outputs_info=[i0, None],
                          n_steps=3)
x = T.itensor3()

x_ = np.arange(8, dtype='int32').reshape(2, 2, 2)
dimshuffle = t.dimshuffle(2, 1, 0)
flatten = dimshuffle.flatten(ndim=2)
y = flatten.T

function = theano.function(inputs=[], outputs=[t, dimshuffle, flatten, y])
for result in function():
    print('-' * 10)
    print(result)
    # print(result.shape)

from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys


def f(i):
    return i + 1


i0 = T.constant(0, dtype="int32")
rng, _ = theano.scan(fn=f,
                     outputs_info=[i0],
                     n_steps=10)
x = T.imatrix()

function = theano.function(inputs=[x], outputs=[x.ravel()])
for result in function(np.arange(9, dtype='int32').reshape(3,3)):
    print('-' * 10)
    print(result)
    print(result.shape)

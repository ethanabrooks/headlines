from __future__ import print_function

import theano
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
import numpy as np
from main import unpickle

params = unpickle('params', dir='/home/ethanbro/headlines/main/')
for name in params:
    print(name)
    print(theano.function([], params[name])())

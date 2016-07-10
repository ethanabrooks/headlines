from __future__ import print_function

import theano.tensor as T
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
import numpy as np
from main import unpickle

batch_size = 2
input_channels = 3
input_rows = 4
input_columns = 5

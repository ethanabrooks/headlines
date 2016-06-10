from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys
from main import pickle, unpickle, evaluate

t, p = map(unpickle, ('targets', 'predictions'))
evaluate(p, t)

# function = theano.function(inputs=[], outputs=[t, dimshuffle, flatten, y])
# for result in function():
#     print('-' * 10)
#     print(result)
    # print(result.shape)

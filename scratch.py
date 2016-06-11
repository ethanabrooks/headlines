from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys

from main import unpickle

articles = unpickle('articles')
titles = unpickle('titles')


print()

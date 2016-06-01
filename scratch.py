from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys

from decimal import Decimal
x =.5010003
exp = int(np.log10(x))
sign = "+"
if x < 1:
    sign = ""
    exp -= 1
print(exp)
coeff = x * 10 ** (-exp)
print(coeff)
print("{:1.2}e{}{}".format(coeff, sign, exp))

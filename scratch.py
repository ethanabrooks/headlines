import theano
import theano.tensor as T
import numpy as np

x = T.constant(np.ones((1024, 40000)))
print(theano.function([], x)())



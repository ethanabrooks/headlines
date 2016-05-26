import theano
import theano.tensor as T
import numpy as np

x = T.constant(np.arange(9).reshape(3, 3))
i = T.constant(np.arange(3))
z = x[:, i]
print(theano.function([], z)())



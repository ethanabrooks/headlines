import numpy as np
import theano
from lasagne.layers import GRULayer, DenseLayer, InputLayer, get_output, EmbeddingLayer
from theano import tensor as T

class Test:
    def __init__(self, i):
        print('test')
        self.x = theano.shared(np.random.random())
        self.i = i


def f(i):
    t = Test(i)
    return [t.i, t.x]

out, _ = theano.scan(fn=f,
                     sequences=T.arange(3))

print(theano.function([], out)())

import keras.backend as K
from keras.layers import Input, Dense, merge, SimpleRNN, Lambda, GRU, Activation, Feedback
from keras.models import Sequential, Model
import numpy as np
import theano
import theano.tensor as T

batch_size = 2
input_dim = 1
midway_dim = 4
output_dim = 1
train_y_dim = 6
# timesteps_i = 5
# timesteps_o = 6



X_batch = np.ones((
    batch_size, input_dim))
# .reshape( batch_size, 1, input_dim)
Y_batch = np.arange(
    batch_size * midway_dim).reshape(
    batch_size, 1, midway_dim)


# input = Input(shape=(input_dim,))
# layer = Dense(output_dim)
# test = Model(input=input, output=layer(input))


def ident(x): return x


test = Sequential()
test.add(Feedback(
    Sequential([SimpleRNN(output_dim,
                          activation=lambda x: x,
                          # input_shape=(1, output_dim,),
                          weights=[np.ones((input_dim, output_dim)),
                                   np.ones((output_dim, output_dim)),
                                   np.ones((output_dim,))],
                          input_shape=()),
                SimpleRNN(output_dim)]
               ), 7, lambda x: x + 1, unroll=False
    # , input_shape=(input_dim,)
))
print(test.predict(X_batch))

import keras.backend as K
from keras.layers import Input, Dense, merge, SimpleRNN, Lambda, GRU, Activation
from keras.models import Sequential, Model
import numpy as np
import theano
import theano.tensor as T

batch_size = 2
input_dim = 3
midway_dim = 4
output_dim = 5
train_y_dim = 6
# timesteps_i = 5
# timesteps_o = 6



X_batch = np.arange(
    batch_size * input_dim).reshape(
    batch_size, input_dim)
Y_batch = np.arange(
    batch_size * midway_dim).reshape(
    batch_size, midway_dim)

encoder = Sequential()
encoder_layer = Dense(midway_dim, input_dim=input_dim)
encoder.add(encoder_layer)

x = Input(shape=(input_dim,))
y = Input(shape=(midway_dim,))


def ladderRecurrence(hidden_state, train_assist):
    # dim_ = Dense(midway_dim)(hidden_state)
    # print(dim_)
    return hidden_state + 2


def ladder(hidden_state, y):
    # if output_size is None:
    #     assert y is not None, "y and output_size can't both be None"
    #     output_size = y.shape[1]
    output, _ = theano.scan(ladderRecurrence,
                            outputs_info=hidden_state,
                            sequences=y)
    return theano.function([hidden_state, y], output)


hidden_state = Input(shape=(midway_dim,))
output = Lambda(ladder, arguments={'y': y})(hidden_state)
output = merge([hidden_state, y])
decoder = Model(input=[hidden_state, y], output=output)

hidden_state = encoder(x)
output = decoder([hidden_state, y])
model = Model(input=[x, y], output=output)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd')
model.train_on_batch([X_batch, Y_batch], Y_batch)

# input1 = Input(shape=(2,))
# input2 = Input(shape=(1,))
# m = merge([input1, input2], mode='concat')
# model3 = Model(input=[input1, input2], output=dense1(m))

# X_batch = np.arange(
#     batch_size * timesteps_i * input_dim).reshape(
#     batch_size, timesteps_i, input_dim)
# Y_batch = np.arange(
#     batch_size * timesteps_o * output_dim).reshape(
#     batch_size, timesteps_o, output_dim)

# model.add(SimpleRNN(output_dim,
#                     input_shape=(timesteps_i, input_dim),
#                     return_sequences=True))

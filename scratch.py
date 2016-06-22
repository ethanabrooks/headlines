from __future__ import print_function

from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
import numpy as np

batch_size = 2
input_dim = 1
midway_dim = 4
output_dim = 1
train_y_dim = 6
timesteps_i = 5
timesteps_o = 6

X_batch = np.ones((
    batch_size, input_dim))
# .reshape( batch_size, 1, input_dim)
Y_batch = np.arange(
    batch_size * midway_dim).reshape(
    batch_size, 1, midway_dim)

dense = Dense(output_dim, input_shape=(input_dim,))
model1 = Sequential([
    dense
])

x = Input(shape=(input_dim,))
f = lambda input: input + 1
# layers = []
# for output in f(x):
#     layers.append(Lambda(lambda _: output)(x))
output = Lambda(lambda _: f(x))(x)
model = Model(input=x, output=output)

print(model.predict(X_batch))

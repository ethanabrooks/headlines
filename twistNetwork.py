from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.layers import GRU, Dense

dim = 4
nclasses = 4
up_net = GRU(dim)
down_net = GRU(dim)
mlp_out = Dense(nclasses)


from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.layers import GRU, Dense
import numpy as np
import seq2seq.models

dim = 4
nclasses = 4
up_net = GRU(dim)
down_net = GRU(dim)
mlp_out = Dense(nclasses)

class FeedbackLayer(Layer):
    def __init__(self, (n_samples, timesteps, input_dim), hidden_dim, depth, **kwargs):
        self.depth = depth
        self.n_samples = n_samples
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def depth_step(self, s_td):

    def time_step(self, state_t0):
        state_td = state_t0
        new_states = []
        for d in range(self.depth):
            new_state, state_td = self.depth_step(state_td)
            new_states.append(new_state)



    def call(self, state_0, train_value=None):
        state_t = state_0
        outputs = []
        for t in range(self.timesteps):
            output, state_t = self.time_step(state_t)
            outputs.append(output)



    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

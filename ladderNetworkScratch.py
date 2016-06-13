from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.layers import GRU, Dense
import numpy as np
from keras.models import Sequential

dim = 4
nclasses = 4
up_net = GRU(dim)
down_net = GRU(dim)
mlp_out = Dense(nclasses)

class LadderNetwork(Layer):
    def __init__(self, hidden_dim, depth, **kwargs):
        self.hidden_dim = hidden_dim
        self.depth = depth
        super(LadderNetwork, self).__init__(**kwargs)

    def build(self, input_shape):  # (n_samples, timesteps, input_dim)
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def time_step(self, inputs_tm1, train_value=None):
        assert len(inputs_tm1) == self.depth

        # traverse upward
        upward_state = None
        rightward_states = []
        for i in range(self.depth):
            upward_state, rightward_state = self.cells[i, 0].call(inputs_tm1[i],
                                                                  [upward_state])
            rightward_states.append(rightward_state)
        output = upward_state

        # traverse downward
        downward_state = upward_state if train_value is None else train_value
        output_states = []
        for i in reversed(range(self.depth)):
            downward_state, output_state = self.cells[i, 1](rightward_states[i],
                                                            [downward_state])
            output_states.append(output_state)
        return output, output_states

    def call(self, state, train_values=None):
        states = [state] + [None] * (self.depth - 1)
        outputs = []
        for t in range(self.timesteps):
            train_value = None if train_values is None else train_values[t]
            output, states = self.time_step(states, train_value)
            outputs.append(output)
        return K.concatenate(outputs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

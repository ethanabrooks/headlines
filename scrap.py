from functools import partial
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell\
    import GRUCell, RNNCell, MultiRNNCell, LSTMCell, BasicRNNCell

sess = tf.Session()
batch_size = 2
memory_size = 3
n_memory_slots = 4
embedding_dim = 5
hidden_size = 6
depth = 2

shapes = {
    'gru_state': (batch_size, embedding_dim),
    'h': (batch_size, hidden_size),
    'M': (batch_size, memory_size, n_memory_slots),
    'w': (batch_size, n_memory_slots, 1),
    'input': (batch_size, hidden_size)
}


def ones_variable(name):
    shape = shapes[name]
    return tf.Variable(np.ones(shape), dtype=tf.float32, name=name)

with tf.Session() as sess:
    M = ones_variable('M')
    w = ones_variable('w')
    tf.initialize_all_variables().run()
    print(sess.run(tf.batch_matmul(M, w)))

# x = tf.zeros([2, 2])
# m = tf.zeros([2, 2])
# g, _ = tf.nn.rnn_cell.BasicRNNCell(2)(x, x)

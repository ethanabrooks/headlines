from functools import partial
import tensorflow as tf
import numpy as np

sess = tf.Session()
batch_size = 2
memory_size = 3
n_memory_slots = 4
x = tf.constant(np.arange(5, dtype='int32'))
equal = tf.equal(x, tf.constant(0, dtype=tf.int32))
titles = np.arange(4).reshape(2, 2)
x = np.pad(titles, ((0, 0), (0, 1)), 'constant', constant_values=0)
target_weights = titles[:, 1:] != 0
target_weights = np.c_[target_weights, np.zeros(titles.shape[0])]
print(target_weights)


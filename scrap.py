from functools import partial
import tensorflow as tf
import numpy as np

sess = tf.Session()
batch_size = 2
memory_size = 3
n_memory_slots = 4
x = tf.constant(np.arange(-8, 8, dtype='float32')) #.reshape(2, 2, 4))
y = tf.nn.l2_normalize(x, dim=0)

print(sess.run([y]))


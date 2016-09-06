from functools import partial
import tensorflow as tf
import numpy as np

sess = tf.Session()
batch_size = 2
memory_size = 3
n_memory_slots = 4
x = tf.constant(np.arange(16, dtype='int32').reshape(2, 2, 4))

print(sess.run(x[:, 1:2, 0]))


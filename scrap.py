from functools import partial
import tensorflow as tf
import numpy as np

sess = tf.Session()
batch_size = 2
memory_size = 3
n_memory_slots = 4
x = tf.constant(np.arange(5, dtype='int32'))
equal = tf.equal(x, tf.constant(0, dtype=tf.int32))
x = tf.constant([2, 6])
y = tf.constant([4, 5])


r = tf.select(tf.less(x, y), x, y)
print(sess.run([r]))

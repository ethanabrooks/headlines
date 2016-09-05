from __future__ import print_function

import os
import pickle
from functools import partial

import numpy as np
import tensorflow as tf

bsize = 2
out_dim = 4
seq_len = 3

x = tf.constant(np.arange(bsize * seq_len * out_dim)
                .reshape(bsize, seq_len, out_dim),
                dtype=tf.float32)
y = tf.constant(np.arange(bsize * seq_len * out_dim)
                .reshape(bsize, seq_len, out_dim),
                dtype=tf.float32)
print(x)
print(y)
tf.Session().run(tf.batch_matmul(x, y, adj_y=True))

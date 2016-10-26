from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell, MultiRNNCell


def cosine_distance(memory, keys):
    """
    :param memory: [batch_size, memory_dim, n_memory_slots]
    :param keys:   [batch_size, memory_dim]
    :return:       [batch_size, n_memory_slots]
    """
    broadcast_keys = tf.expand_dims(keys, dim=2)

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x), reduction_indices=1, keep_dims=True))

    norms = map(norm, [memory, broadcast_keys])  # [batch_size, n_memory_slots]
    dot_product = tf.squeeze(tf.batch_matmul(broadcast_keys,
                                             memory,
                                             adj_x=True))  # [batch_size, n_memory_slots]
    norms_product = tf.squeeze(tf.nn.softplus(tf.mul(*norms)))
    return dot_product / norms_product


def gather(tensor, indices, axis=2, ndim=3):
    assert axis < ndim
    perm = np.arange(ndim)
    perm[0] = axis
    perm[axis] = 0
    return tf.transpose(tf.gather(tf.transpose(tensor, perm), indices), perm)


class ntmCell(RNNCell):
    def __init__(self,
                 go_code,
                 embedding_dim,
                 hidden_size,
                 memory_dim,
                 n_memory_slots,
                 n_classes):

        randoms = {
            # attr: shape
            # 'emb': (num_embeddings + 1, embedding_dim),
            'Wg': (hidden_size, n_memory_slots),
            'Wk': (hidden_size, memory_dim),
            'Wb': (hidden_size, 1),
            'Wv': (hidden_size, memory_dim),
            'We': (hidden_size, n_memory_slots),
            'Wx': (embedding_dim, hidden_size),
            'Wc': (memory_dim, hidden_size),
            'W': (hidden_size, n_classes),
        }

        zeros = {
            # attr: shape
            'bg': n_memory_slots,
            'bk': memory_dim,
            'bb': 1,
            'bv': memory_dim,
            'be': n_memory_slots,
            'bh': hidden_size,
            'b': n_classes,
        }

        def random_shared(name):
            shape = randoms[name]
            return tf.Variable(0.2 * np.random.normal(size=shape),
                               dtype=tf.float32, name=name)

        def zeros_shared(name):
            shape = zeros[name]
            return tf.Variable(np.zeros(shape),
                               dtype=tf.float32, name=name)

        for key in randoms:
            # create an attribute with associated shape and random values
            setattr(self, key, random_shared(key))

        for key in zeros:
            # create an attribute with associated shape and values equal to 0
            setattr(self, key, zeros_shared(key))

        self.names = randoms.keys() + zeros.keys()
        self.is_article = tf.constant(True)
        self.go_code = go_code

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.memory_dim = memory_dim
        self.n_memory_slots = n_memory_slots
        self.i = 0

    @property
    def output_size(self):
        return self.output_size

    @property
    def state_size(self):
        return self.hidden_size + \
               self.n_memory_slots + \
               self.memory_dim * self.n_memory_slots

    def __call__(self, x, state, name=None):
        """
        :param x: [batch_size, hidden_size]
        :param state: [1, state_size]
        :return:
        """
        # PARSE STATE VARIABLES

        batch_size = tf.size(state) / self.state_size

        w_start = batch_size * self.hidden_size
        h_tm1 = state[:w_start]
        h_tm1 = tf.reshape(h_tm1, (-1, self.hidden_size))
        # [batch_size, hidden_size]

        M_start = batch_size * (self.hidden_size + self.n_memory_slots)
        w_tm1 = state[w_start: M_start]
        w_tm1 = tf.reshape(w_tm1, (-1, self.n_memory_slots))
        # [batch_size, n_memory_slots]

        M = state[:-M_start]
        M = tf.reshape(M, (-1, self.memory_dim, self.n_memory_slots))
        # [batch_size, memory_dim, n_memory_slots]

        self.is_article = tf.cond(
            # if the first column of inputs is the go code
            tf.equal(x[0, 0], self.go_code),
            lambda: tf.logical_not(self.is_article),  # flip the value of self.is_article
            lambda: self.is_article  # otherwise leave it alone
        )

        # eqn 15
        c = tf.squeeze(tf.batch_matmul(M, tf.expand_dims(w_tm1, dim=2)))
        # [batch_size, memory_dim]

        # EXTERNAL MEMORY READ
        g = tf.sigmoid(tf.matmul(h_tm1, self.Wg) + self.bg)
        # [batch_size, memory_dim]

        # eqn 11
        k = tf.matmul(h_tm1, self.Wk) + self.bk
        # [batch_size, memory_dim]

        # eqn 13
        beta = tf.matmul(h_tm1, self.Wb) + self.bb
        beta = tf.nn.softplus(beta)
        # [batch_size, 1]

        # eqn 12
        w_hat = tf.nn.softmax(beta * cosine_distance(M, k))
        # [batch_size, n_memory_slots]

        # eqn 14
        w_t = (1 - g) * w_tm1 + g * w_hat
        # [batch_size, n_memory_slots]

        # MODEL INPUT AND OUTPUT

        n_article_slots = self.n_memory_slots / 2
        read_idxs = tf.cond(self.is_article,
                            lambda: tf.range(0, n_article_slots),
                            lambda: tf.range(0, self.n_memory_slots))

        c = gather(c, indices=read_idxs, axis=1, ndim=2)
        Wc = gather(self.Wc, indices=read_idxs, axis=0, ndim=2)

        # eqn 9
        h_t = tf.nn.sigmoid(tf.matmul(x, self.Wx) + tf.matmul(c, Wc) + self.bh)
        # [batch_size, hidden_size]

        # eqn 10
        y = tf.nn.softmax(tf.matmul(h_t, self.W) + self.b)
        # [batch_size, nclasses]

        # EXTERNAL MEMORY UPDATE
        # eqn 17
        e = tf.nn.sigmoid(tf.matmul(h_t, self.We) + self.be)
        # [batch_size, n_memory_slots]

        f = w_t * e
        # [batch_size, n_memory_slots]

        # eqn 16
        v = tf.nn.tanh(tf.matmul(h_t, self.Wv) + self.bv)

        # [batch_size, memory_dim]

        def broadcast(x, dim, size):
            multiples = [1, 1, 1]
            multiples[dim] = size
            return tf.tile(tf.expand_dims(x, dim), multiples)

        f = broadcast(f, 1, self.memory_dim)
        # [batch_size, memory_dim, n_memory_slots]

        u = broadcast(w_t, 1, 1)
        # [batch_size, 1, n_memory_slots]

        v = broadcast(v, 2, 1)
        # [batch_size, memory_dim, 1]

        # eqn 19
        M_update = M * (1 - f) + tf.batch_matmul(v, u) * f  # [batch_size, memory_dim, mem]

        # determine whether to update article or title
        M_article = tf.cond(self.is_article, lambda: M_update, lambda: M)
        M_title = tf.cond(self.is_article, lambda: M, lambda: M_update)

        article_idxs = tf.range(0, n_article_slots)
        title_idxs = tf.range(n_article_slots, self.n_memory_slots)

        M_article = gather(M_article, indices=article_idxs, axis=2, ndim=3)
        M_title = gather(M_title, indices=title_idxs, axis=2, ndim=3)

        # join updated with non-updated subtensors in M
        M = tf.concat(concat_dim=2, values=[M_article, M_title])

        h_t = tf.reshape(h_t, (-1,))
        w_t = tf.reshape(w_t, (-1,))
        M = tf.reshape(M, (-1,))
        return y, tf.concat(0, [h_t, w_t, M])


if __name__ == '__main__':
    with tf.Session() as sess:
        batch_size = 3
        hidden_size = 2
        embedding_dim = 5
        memory_dim = 3
        n_memory_slots = 4
        depth = 1
        n_classes = 12

        x = tf.constant(np.random.uniform(high=batch_size * embedding_dim,
                                          size=(batch_size, embedding_dim)) * np.sqrt(3),
                        dtype=tf.float32)

        cell = ntmCell(go_code=1,
                       n_classes=n_classes,
                       embedding_dim=embedding_dim,
                       hidden_size=hidden_size,
                       memory_dim=memory_dim,
                       n_memory_slots=n_memory_slots)

        # state_shapes = {
        #     'gru_state': (batch_size, embedding_dim),
        #     'h': (batch_size, hidden_size ),
        #     'M': (batch_size, n_memory_slots * memory_dim),
        #     'w': (batch_size, n_memory_slots),
        # }

        # def zeros_variable(name):
        #     shape = state_shapes[name]
        #     return tf.Variable(np.zeros(shape), name=name)

        states_dim = (hidden_size + n_memory_slots + n_memory_slots * memory_dim) * batch_size
        states = tf.Variable(np.zeros(states_dim), dtype=tf.float32)
        output = cell(x, states)
        tf.initialize_all_variables().run()
        result = sess.run(output)


        def print_lists(result):
            if type(result) == list or type(result) == tuple:
                for x in result:
                    print('-' * 10)
                    print_lists(x)
            else:
                print(result)
                print(result.shape)


        print_lists(result)

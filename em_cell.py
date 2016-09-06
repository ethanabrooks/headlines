from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell


def cosine_distance(memory, keys):
    """
    :param memory: [instances, memory_size, n_memory_slots]
    :param keys:   [instances, memory_size]
    :return:       [instances, n_memory_slots]
    """
    broadcast_keys = tf.expand_dims(keys, dim=2)

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.nn.l2_normalize(x, dim=1),
                                     reduction_indices=1))

    norms = map(norm, [memory, broadcast_keys])  # [instances, n_memory_slots]
    dot_prod = tf.squeeze(tf.batch_matmul(broadcast_keys,
                                          memory,
                                          adj_x=True))  # [instances, n_memory_slots]
    softplus = tf.nn.softplus(tf.mul(*norms))
    return dot_prod, softplus, dot_prod / softplus


def gather(tensor, indices, axis=2, ndim=3):
    assert axis < ndim
    perm = np.arange(ndim)
    perm[0] = axis
    perm[axis] = 0
    return tf.transpose(tf.gather(tf.transpose(tensor, perm), indices), perm)


class EMCell(RNNCell):
    def __init__(self,
                 go_code,
                 depth,
                 batch_size,
                 embedding_dim,
                 hidden_size,
                 memory_size,
                 n_memory_slots,
                 n_classes,
                 load_dir=None):

        randoms = {
            # attr: shape
            # 'emb': (num_embeddings + 1, embedding_dim),
            'Wg': (embedding_dim, n_memory_slots),
            'Wk': (hidden_size, memory_size),
            'Wb': (hidden_size, 1),
            'Wv': (hidden_size, memory_size),
            'We': (hidden_size, n_memory_slots),
            'Wx': (embedding_dim, hidden_size),
            'Wh': (memory_size, hidden_size),
            'W': (hidden_size, n_classes),
        }

        zeros = {
            # attr: shape
            'gru_state': (batch_size, embedding_dim),
            'h': (batch_size, hidden_size),
            'M': (batch_size, n_memory_slots * memory_size),
            'a': (batch_size, n_memory_slots),
            'bg': n_memory_slots,
            'bk': memory_size,
            'bb': 1,
            'bv': memory_size,
            'be': n_memory_slots,
            'bh': hidden_size,
            'b': n_classes,
        }

        for l in range(depth):
            randoms['gru' + str(l)] = (1, embedding_dim)

        def random_shared(name):
            shape = randoms[name]
            return tf.Variable(
                np.arange(np.prod(shape)).reshape(shape),
                dtype=tf.float32,
                name=name)
            # return tf.Variable(
            #     0.2 * np.random.normal(size=shape),
            #     dtype=tf.float32,
            #     name=name)

        def zeros_shared(name):
            shape = zeros[name]
            return tf.Variable(
                0.2 * np.random.normal(size=shape),
                dtype=tf.float32,
                name=name)

        for key in randoms:
            # create an attribute with associated shape and random values
            setattr(self, key, random_shared(key))

        for key in zeros:
            # create an attribute with associated shape and values equal to 0
            setattr(self, key, zeros_shared(key))

        self.names = randoms.keys() + zeros.keys()
        self.gru = GRUCell(embedding_dim)
        self.is_article = tf.constant(True)
        self.go_code = go_code

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.memory_size = memory_size
        self.n_memory_slots = n_memory_slots

    @property
    def output_size(self):
        return self.output_size

    @property
    def state_size(self):
        return (self.gru.state_size,
                self.hidden_size,
                self.n_memory_slots,
                self.n_memory_slots * self.memory_size)

    def __call__(self, inputs, (gru_state, h_tm1, w_tm1, M), name=None):
        """
        :param inputs: [batch_size, hidden_size]
        :return:
        """

        # batch_size = 3
        # dim = 1
        # articles1 = np.random.r(batch_size * dim, dtype='int32') \
        #     .reshape(batch_size, seq_len1)  # np.load(dir + "article.npy")
        # with tf.Session() as sess:
        #     print(sess.run(M, feed_dict={inputs: articles1}))

        M = tf.reshape(M, (-1, self.memory_size, self.n_memory_slots))
        # [instances, memory_size, n_memory_slots]

        self.is_article = tf.cond(
            # if the first column of inputs is the go code
            tf.equal(inputs[0, 0], self.go_code),
            lambda: tf.logical_not(self.is_article),  # flip the value of self.is_article
            lambda: self.is_article  # otherwise leave it alone
        )

        gru_outputs, gru_state = self.gru(inputs, gru_state)
        # [batch_size, embedding_dim]

        # eqn 15
        c = tf.squeeze(tf.batch_matmul(M, tf.expand_dims(w_tm1, dim=2)))
        # [instances, memory_size]

        # EXTERNAL MEMORY READ
        g = tf.sigmoid(tf.matmul(gru_outputs, self.Wg) + self.bg)
        # [instances, memory_size]

        # eqn 11
        k = tf.matmul(h_tm1, self.Wk) + self.bk
        # [instances, memory_size]

        # eqn 13
        beta = tf.matmul(h_tm1, self.Wb) + self.bb
        beta = tf.nn.softplus(beta)
        # [instances, 1]

        # eqn 12
        dot_prod, softplus, distance = cosine_distance(M, k)
        w_hat = tf.nn.softmax(beta * distance)
        # [instances, n_memory_slots]

        # eqn 14
        w_t = (1 - g) * w_tm1 + g * w_hat
        # [instances, n_memory_slots]

        # MODEL INPUT AND OUTPUT

        n_article_slots = self.n_memory_slots / 2
        read_idxs = tf.cond(self.is_article,
                            lambda: tf.range(0, n_article_slots),
                            lambda: tf.range(0, self.n_memory_slots))

        c = gather(c, indices=read_idxs, axis=1, ndim=2)
        Wh = gather(self.Wh, indices=read_idxs, axis=0, ndim=2)
        # eqn 9
        h_t = tf.matmul(c, Wh) + tf.matmul(gru_outputs, self.Wx) + self.bh
        # [instances, hidden_size]

        # eqn 10
        y = tf.nn.softmax(tf.matmul(h_t, self.W) + self.b)
        # [instances, nclasses]

        # EXTERNAL MEMORY UPDATE
        # eqn 17
        e = tf.nn.sigmoid(tf.matmul(h_t, self.We) + self.be)
        # [instances, n_memory_slots]

        f = w_t * e
        # [instances, n_memory_slots]

        # eqn 16
        v = tf.nn.tanh(tf.matmul(h_t, self.Wv) + self.bv)

        # [instances, memory_size]

        def broadcast(x, dim, size):
            multiples = [1, 1, 1]
            multiples[dim] = size
            return tf.tile(tf.expand_dims(x, dim), multiples)

        f = broadcast(f, 1, self.memory_size)
        # [instances, memory_size, n_memory_slots]

        u = broadcast(w_t, 1, 1)
        # [instances, 1, n_memory_slots]

        v = broadcast(v, 2, 1)
        # [instances, memory_size, 1]

        # eqn 19
        M_update = M * (1 - f) + tf.batch_matmul(v, u) * f  # [instances, memory_size, mem]

        # determine whether to update article or title
        M_article = tf.cond(self.is_article, lambda: M_update, lambda: M)
        M_title = tf.cond(self.is_article, lambda: M, lambda: M_update)

        article_idxs = tf.range(0, n_article_slots)
        title_idxs = tf.range(n_article_slots, self.n_memory_slots)

        M_article = gather(M_article, indices=article_idxs, axis=2, ndim=3)
        M_title = gather(M_title, indices=title_idxs, axis=2, ndim=3)

        # join updated with non-updated subtensors in M
        # M = tf.concat(concat_dim=2, values=[M_article, M_title])
        return y, (beta, k, M, distance, w_hat)
        return y, (gru_state, h_t, w_t, M)


if __name__ == '__main__':
    with tf.Session() as sess:  # , tf.variable_scope("", reuse=True):
        # dir = "train/5-3/"
        batch_size = 3
        hidden_size = 2
        embedding_dim = 5
        memory_size = 3
        n_memory_slots = 4

        x = tf.constant(np.random.uniform(high=batch_size * hidden_size,
                        size=(batch_size, hidden_size)) * np.sqrt(3), dtype=tf.float32)

        cell = EMCell(go_code=1,
                      depth=1,
                      batch_size=batch_size,
                      n_classes=12,
                      embedding_dim=embedding_dim,
                      hidden_size=hidden_size,
                      memory_size=memory_size,
                      n_memory_slots=n_memory_slots)
        output = cell(x, (cell.gru_state, cell.h, cell.a, cell.M))
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

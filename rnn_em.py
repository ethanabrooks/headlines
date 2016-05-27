from __future__ import print_function

import os
import pickle
from functools import partial

import lasagne
import numpy
import theano
from theano import tensor as T

int32 = 'int32'


def cosine_dist(tensor, matrix):
    """
    Along axis 1 for both inputs.
    Assumes dimensions 0 and 1 are equal
    """
    matrix_norm = T.shape_padright(matrix.norm(2, axis=1))
    tensor_norm = tensor.norm(2, axis=1)
    return T.batched_dot(matrix, tensor) / (matrix_norm * tensor_norm)


# noinspection PyPep8Naming,PyUnresolvedReferences
class Model(object):
    def __init__(self,
                 hidden_size=100,
                 nclasses=3,
                 num_embeddings=11359,
                 embedding_dim=100,
                 window_size=1,  # TODO: do we want some kind of window?
                 memory_size=40,
                 n_memory_slots=8,
                 go_code=0):

        articles, titles = T.imatrices('articles', 'titles')
        n_article_slots = int(n_memory_slots / 2)  # TODO derive this from an arg
        n_title_slots = n_memory_slots - n_article_slots
        n_instances = articles.shape[0]

        self.window_size = window_size

        randoms = {
            # attr: shape
            'emb': (num_embeddings + 1, embedding_dim),
            'Wg_a': (window_size * embedding_dim, n_article_slots),
            'Wg_t': (window_size * embedding_dim, n_title_slots),
            'Wk': (hidden_size, memory_size),
            'Wb': (hidden_size, 1),
            'Wv': (hidden_size, memory_size),
            'We_a': (hidden_size, n_article_slots),
            'We_t': (hidden_size, n_title_slots),
            'Wx': (window_size * embedding_dim, hidden_size),
            'Wh': (memory_size, hidden_size),
            'W': (hidden_size, nclasses),
            'h0': hidden_size,
            'w_a': (n_article_slots,),
            'w_t': (n_title_slots,),
            'M_a': (memory_size, n_article_slots),
            'M_t': (memory_size, n_title_slots)
        }

        zeros = {
            # attr: shape
            'bg_a': n_article_slots,
            'bg_t': n_title_slots,
            'bk': memory_size,
            'bb': 1,
            'bv': memory_size,
            'be_a': n_article_slots,
            'be_t': n_title_slots,
            'bh': hidden_size,
            'b': nclasses,
        }

        def random_shared(shape):
            return theano.shared(
                0.2 * numpy.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX))

        def zeros_shared(shape):
            return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))

        for key in randoms:
            # create an attribute with associated shape and random values
            try:
                setattr(self, key, random_shared(randoms[key]))
            except MemoryError:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(key)

        for key in zeros:
            # create an attribute with associated shape and values = 0
            setattr(self, key, zeros_shared(zeros[key]))

        self.names = randoms.keys() + zeros.keys()
        scan_vars = 'h0 w_a M_a w_t M_t'.split()

        def repeat_for_each_instance(param):
            """ repeat param along new axis once for each instance """
            return T.repeat(T.shape_padleft(param), repeats=n_instances, axis=0)

        for key in scan_vars:
            setattr(self, key, repeat_for_each_instance(self.__getattribute__(key)))
            self.names.remove(key)

        self.params = [eval('self.' + name) for name in self.names]

        def recurrence(i, h_im1, w_a, M_a, w_t=None, M_t=None, is_training=True, is_article=True):
            """
            notes
            Headers from paper in all caps
            mem = n_article slots if is_article else n_title_slots

            :param i: center index of sliding window
            :param h_im1: h_{t-1} (hidden state)
            :param w_a: attention weights for article memory
            :param M_a: article memory
            :param w_t: attention weights for titles memory
            :param M_t: titles memory
            :param is_training:
            :param is_article: we use different parts of memory when working with a article
            :return: [y = model outputs,
                      i + 1 = increment index,
                      h w, M (see above)]
            """
            i_type = T.iscalar if is_article or is_training else T.ivector
            assert i.type == i_type

            if not is_article:
                assert w_t is not None and M_t is not None

            word_idxs = i
            if is_article or is_training:
                # get representation of word window
                document = articles if is_article else titles  # [instances, bucket_width]
                word_idxs = document[i]  # [instances, 1]
            x_i = self.emb[word_idxs].flatten(ndim=2)  # [instances, embedding_dim]

            if is_article:
                M_read = M_a  # [instances, memory_size, n_article_slots]
                w_read = w_a  # [instances, n_article_slots]
            else:
                M_read = T.concatenate([M_a, M_t], axis=2)  # [instances, memory_size, n_title_slots]
                w_read = T.concatenate([w_a, w_t], axis=1)  # [instances, n_title_slots]

            # eqn 15
            c = T.batched_dot(M_read, w_read)  # [instances, memory_size]

            # EXTERNAL MEMORY READ
            def get_attention(Wg, bg, M, w):
                g = T.nnet.sigmoid(T.dot(x_i, Wg) + bg)  # [instances, mem]

                # eqn 11
                k = T.dot(h_im1, self.Wk) + self.bk  # [instances, memory_size]

                # eqn 13
                beta = T.dot(h_im1, self.Wb) + self.bb
                beta = T.log(1 + T.exp(beta))
                beta = T.addbroadcast(beta, 1)  # [instances, 1]

                # eqn 12
                w_hat = T.nnet.softmax(beta * cosine_dist(M, k))

                # eqn 14
                return (1 - g) * w + g * w_hat  # [instances, mem]

            w_a = get_attention(self.Wg_a, self.bg_a, M_a, w_a)  # [instances, n_article_slots]
            if not is_article:
                w_t = get_attention(self.Wg_t, self.bg_t, M_t, w_t)  # [instances, n_title_slots]

            # MODEL INPUT AND OUTPUT
            # eqn 9
            h = T.dot(x_i, self.Wx) + T.dot(c, self.Wh) + self.bh  # [instances, hidden_size]

            # eqn 10
            y = T.nnet.softmax(T.dot(h, self.W) + self.b)  # [instances, nclasses]

            # EXTERNAL MEMORY UPDATE
            def update_memory(We, be, w_update, M_update):
                # eqn 17
                e = T.nnet.sigmoid(T.dot(h_im1, We) + be)  # [instances, mem]
                f = 1. - w_update * e  # [instances, mem]

                # eqn 16
                v = T.dot(h, self.Wv) + self.bv  # [instances, memory_size]

                # need to add broadcast layers for memory update
                f = f.dimshuffle(0, 'x', 1)  # [instances, 1, mem]
                u = w_update.dimshuffle(0, 'x', 1)  # [instances, 1, mem]
                v = v.dimshuffle(0, 1, 'x')  # [instances, memory_size, 1]

                # eqn 19
                return M_update * f + T.batched_dot(v, u) * (1 - f)  # [instances, memory_size, mem]

            M_a = update_memory(self.We_a, self.be_a, w_a, M_a)
            attention_and_memory = [w_a, M_a]
            if not is_article:
                M_t = update_memory(self.We_t, self.be_t, w_t, M_t)
                attention_and_memory += [w_t, M_t]

            y_max = y.argmax(axis=1).astype(int32)
            next_idxs = i + 1 if is_training or is_article else y_max
            return [y, y_max, next_idxs, h] + attention_and_memory

        read_article = partial(recurrence, is_article=True)
        i0 = T.constant(0, dtype=int32)
        outputs_info = [None, None, i0, self.h0, self.w_a, self.M_a]
        [_, _, _, h, w, M], _ = theano.scan(fn=read_article,
                                            outputs_info=outputs_info,
                                            n_steps=articles.shape[1],
                                            name='read_scan')

        produce_title = partial(recurrence, is_training=True, is_article=False)
        outputs_info[3:] = [param[-1, :, :] for param in (h, w, M)]
        outputs_info.extend([self.w_t, self.M_t])
        [y, y_max, _, _, _, _, _, _], _ = theano.scan(fn=produce_title,
                                                      outputs_info=outputs_info,
                                                      n_steps=titles.shape[1],
                                                      name='train_scan')

        # loss and updates
        y = y.dimshuffle(2, 1, 0).flatten(ndim=2).T
        y_true = titles.ravel()
        counts = T.extra_ops.bincount(y_true, assert_nonneg=True)
        weights = 1.0 / (counts[y_true] + 1) * T.neq(y_true, 0)
        losses = T.nnet.categorical_crossentropy(y, y_true)
        loss = lasagne.objectives.aggregate(losses, weights, mode='sum')
        updates = lasagne.updates.adadelta(loss, self.params)

        self.learn = theano.function(inputs=[articles, titles],
                                     outputs=[y_max, loss],
                                     updates=updates,
                                     allow_input_downcast=True)

        produce_title_test = partial(recurrence, is_training=False, is_article=False)
        outputs_info[2] = T.zeros([n_instances], dtype=int32) + go_code
        [_, y_max, _, _, _, _, _, _], _ = theano.scan(fn=produce_title_test,
                                                      outputs_info=outputs_info,
                                                      n_steps=titles.shape[1],
                                                      name='test_scan')

        self.infer = theano.function(inputs=[articles, titles],
                                     outputs=y_max)

        normalized_embeddings = self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: normalized_embeddings})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


if __name__ == '__main__':
    rnn = Model()
    articles = pickle.load("articles.pkl")
    titles = pickle.load("titles.pkl")
    for result in rnn.test(articles, titles):
        print('-' * 10)
        print(result)
    rnn.normalize()

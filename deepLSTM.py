from __future__ import print_function

import os
import pickle
from functools import partial

import numpy as np
import tensorflow as tf

int32 = 'int32'

optimizers = {
    1: tf.train.GradientDescentOptimizer(.1),
    2: tf.train.AdadeltaOptimizer(),
    3: tf.train.AdagradOptimizer(.01),
    4: tf.train.MomentumOptimizer(.1, 1.),
    5: tf.train.AdamOptimizer(.1),
    6: tf.train.FtrlOptimizer(.1),
    7: tf.train.RMSPropOptimizer(.1)
}

optimizer = optimizers[3]


# noinspection PyPep8Naming,PyUnresolvedReferences
class Model(object):
    def __init__(self,
                 hidden_size=100,
                 nclasses=73,
                 num_embeddings=11359,
                 embedding_dim=100,
                 window_size=1,  # TODO: do we want some kind of window?
                 memory_size=40,
                 n_memory_slots=8,
                 go_code=1):

        self.buckets = dict()

    def encode(self, articles):
        outputs_per_token = self.deepLSTM(self.embed(articles))
        attention = map(self.pay_attention, outputs_per_token)
        outputs_per_token, attention = map(tf.pack, outputs_per_token, attention)
        return tf.matmul(outputs_per_token, attention)

    def decode_train(self, articles_summary, titles):
        return titles  # TODO

    def compare(self, predictions, titles):
        indices = tf.range(1, titles.shape[1])
        titles_without_go = tf.transpose(tf.gather(tf.transpose(titles), indices))
        return tf.nn.softmax_cross_entropy_with_logits(predictions, titles_without_go)

    def learn(self, articles_arr, titles_arr):
        assert articles_arr.shape[0] == titles_arr.shape[0]
        key = articles_arr.shape, titles_arr.shape
        if key not in self.buckets:
            self.buckets[key] = (tf.placeholder(tf.int32, shape)
                                 for shape in key)
        articles, titles = self.buckets[key]
        articles_summary = self.encode(articles)
        predictions = self.decode_train(articles_summary, titles)
        loss = self.compare(predictions, titles)
        train_op = optimizer.minimize(loss)
        with tf.Session() as sess:
            _, summary, loss_value, train_outputs = sess.run(
                [train_op, train_summary, loss, outputs], feed_dict)


        articles, titles = T.imatrices('articles', 'titles')
        n_article_slots = int(n_memory_slots / 2)  # TODO derive this from an arg
        n_title_slots = n_memory_slots - n_article_slots
        n_instances = articles.shape[0]

if __name__ == '__main__':
    articles = np.load("articles.npy")
    titles = np.load("titles.npy")
    rnn = Model()
    # rnn.load('.')
    for result in rnn.test(articles, titles):
        pass
        print('-' * 10)
        print(result)
        print(result.shape)

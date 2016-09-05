from em_cell import EMCell
from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell


class Model(object):
    def __init__(self, **kwargs):
        self.embedding_dim = None
        self.articles = None
        self.titles = None

        self.__dict__.update(kwargs)
        self.num_embeddings = n_classes
        self.cell = EMCell(**kwargs)
        self.learn = partial(self.infer, is_testing=False)
        self.predict = partial(self.infer, is_testing=True)

    def infer(self, articles, titles, is_testing):
        assert (articles.shape[0] == titles.shape[0])

        with tf.Session() as sess, tf.variable_scope('learn') as scope:
            # article_tensors, title_tensors = zip([(tf.placeholder(tf.float32, tensor.shape)
            #                                        for tensor in tensors)
            #                                       for tensors in zip(articles, titles)])

            # buckets = [(5, 5)]  # TODO

            # check if we are on an old batch
            try:
                assert (self.articles._shape == articles.shape)
                assert (self.titles._shape == titles.shape)

            # if not, allocate new placeholders
            except (AssertionError, AttributeError):
                self.articles = tf.placeholder(tf.int32, articles.shape)
                self.titles = tf.placeholder(tf.int32, titles.shape)

            feed_dict = {self.articles: articles,
                         self.titles: titles}

            article_inputs, title_inputs = map(partial(tf.unpack, axis=1),
                                               feed_dict.keys())

            def seq2seq_function():
                return seq2seq.embedding_rnn_seq2seq(
                    article_inputs, title_inputs, self.cell,
                    self.num_embeddings, self.num_embeddings, self.embedding_dim,
                    feed_previous=is_testing)

            # first call to the function: create the GRU state for the first time
            try:
                predictions, _ = seq2seq_function()

            # afterwards, we reuse the previously created state
            except ValueError:
                scope.reuse_variables()
                predictions, _ = seq2seq_function()

            weights = [tf.select(tf.equal(title, 0),
                                 tf.zeros_like(title, dtype=tf.float32),
                                 tf.ones_like(title, dtype=tf.float32))
                       for title in title_inputs]
            loss = seq2seq.sequence_loss(predictions, title_inputs, weights)
            train_op = tf.train.AdadeltaOptimizer().minimize(loss)


            tf.initialize_all_variables().run()
            return sess.run(train_op, feed_dict)


if __name__ == '__main__':
    dir = "train/5-3/"
    batch_size = 3
    seq_len1 = 5
    seq_len2 = 6
    hidden_size = 2
    embedding_dim = 5
    memory_size = 7
    n_memory_slots = 2
    n_classes = batch_size * seq_len1 * hidden_size
    articles1 = np.arange(batch_size * seq_len1, dtype='int32') \
        .reshape(batch_size, seq_len1)  # np.load(dir + "article.npy")
    titles1 = np.arange(batch_size * seq_len1, dtype='int32') \
        .reshape(batch_size, seq_len1)  # np.load(dir + "title.npy")
    articles2 = np.arange(batch_size * seq_len2, dtype='int32')\
        .reshape(batch_size, seq_len2)  # np.load(dir + "article.npy")
    titles2 = np.arange(batch_size * seq_len2, dtype='int32')\
        .reshape(batch_size, seq_len2)  # np.load(dir + "title.npy")
    rnn = Model(go_code=1,
                depth=1,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                memory_size=memory_size,
                n_memory_slots=n_memory_slots,
                n_classes=n_classes)
    # rnn.load('main')
    # rnn.print_params()
    output = rnn.learn(articles1, titles1)
    print('TEST')
    output = rnn.learn(articles2, titles2)
    print('TEST')
    if type(output) == tuple or type(output) == list:
        for result in output:
            print('-' * 10)
            print(result)
        try:
            print(result.shape)
        except AttributeError:
            pass
    else:
        print('-' * 10)
        print(output)
        try:
            print(output.shape)
        except AttributeError:
            pass

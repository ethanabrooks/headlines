from pprint import pprint

from em_cell import EMCell
from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell


class Model(object):
    def __init__(self, session, articles, titles, **kwargs):
        self.session = session
        self.__dict__.update(kwargs)

        # TODO
        self.is_testing = True
        self.learn = partial(self.infer, is_testing=False)
        self.predict = partial(self.infer, is_testing=True)

        self.bucket_sizes = sorted([(article.shape[1], title.shape[1])
                                    for (article, title) in zip(articles, titles)],
                                   key=lambda size: size[0])

        max_sizes = [self.bucket_sizes[-1][0],
                     max(self.bucket_sizes, key=lambda size: size[1])[1]]

        def collect_input(from_encoder, name, add_one=False, dtype=tf.int32):
            return [tf.placeholder(dtype, shape=[None], name=name + str(i))
                    for i in xrange(max_sizes[from_encoder] + add_one)]

        # Feeds for inputs.
        self.encoder_inputs = collect_input(from_encoder=True, name='encoder')
        self.decoder_inputs = collect_input(from_encoder=False, name='decoder', add_one=True)
        self.target_weights = collect_input(from_encoder=False, name='weight',
                                            dtype=tf.float32, add_one=True)

        # Our targets are decoder inputs shifted by one.
        targets = self.decoder_inputs[1:]

        def seq2seq_function(encoder_input, decoder_input):
            return seq2seq.embedding_rnn_seq2seq(
                encoder_input, decoder_input, EMCell(**kwargs),
                self.n_classes, self.n_classes, self.embedding_dim,
                feed_previous=self.is_testing)

        self.outputs, losses = seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs[:-1], targets,
            self.target_weights, self.bucket_sizes, seq2seq_function
        )
        self.train_ops = map(tf.train.AdadeltaOptimizer().minimize, losses)

        tf.initialize_all_variables().run()

    def infer(self, articles, titles, is_testing):
        assert (articles.shape[0] == titles.shape[0])

        def slack(i):
            sizes = self.bucket_sizes[i]
            return sizes[0] - articles.shape[1] + sizes[1] - titles.shape[1]

        ids_of_buckets_that_fit = filter(lambda i: slack(i) >= 0,
                                         range(len(self.bucket_sizes)))
        bucket_id = min(ids_of_buckets_that_fit, key=slack)

        encoder_size, decoder_size = self.bucket_sizes[bucket_id]
        target_weights = titles[:, 1:] != 0
        target_weights = np.c_[target_weights, np.zeros(titles.shape[0])]

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for src, dest in [(articles, self.encoder_inputs),
                          (titles, self.decoder_inputs),
                          (target_weights, self.target_weights)]:
            for l in xrange(src.shape[1]):
                input_feed[dest[l].name] = src[:, l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([titles.shape[0]], dtype=np.int32)

        output_feed = [self.outputs[bucket_id], self.train_ops[bucket_id]]
        outputs, _ = sess.run(output_feed, input_feed)
        return outputs


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
    articles2 = np.arange(batch_size * seq_len2, dtype='int32') \
        .reshape(batch_size, seq_len2)  # np.load(dir + "article.npy")
    titles2 = np.arange(batch_size * seq_len2, dtype='int32') \
        .reshape(batch_size, seq_len2)  # np.load(dir + "title.npy")

    with tf.Session() as sess, tf.variable_scope('learn') as scope:
        rnn = Model(sess, [articles1, articles2], [titles1, titles2],
                    go_code=1,
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
        output = rnn.learn(articles2, titles2)
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

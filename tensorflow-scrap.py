from pprint import pprint

from em_cell import EMCell
from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell


class Model(object):
    def __init__(self, session, bucket_sizes, save_dir, testing, **kwargs):
        self.session = session
        self.testing = testing  # TODO
        self.bucket_sizes = bucket_sizes
        self.__dict__.update(kwargs)

        max_sizes = [self.bucket_sizes[-1][0],
                     max(self.bucket_sizes, key=lambda size: size[1])[1]]

        def make_placeholders(from_encoder, name, extend=0, dtype=tf.int32):
            return [tf.placeholder(dtype, shape=[None], name=name + str(i))
                    for i in xrange(max_sizes[from_encoder] + extend)]

        # Feeds for inputs.
        self.encoder_inputs = make_placeholders(from_encoder=True, name='encoder')
        self.decoder_inputs = make_placeholders(from_encoder=False, name='decoder', extend=1)
        self.target_weights = make_placeholders(from_encoder=False, name='weight',
                                                dtype=tf.float32, extend=1)

        # Our targets are decoder inputs shifted by one.
        targets = self.decoder_inputs[1:]

        def seq2seq_function(encoder_input, decoder_input):
            return seq2seq.embedding_rnn_seq2seq(
                encoder_input, decoder_input, EMCell(**kwargs),
                self.n_classes, self.n_classes, self.embedding_dim,
                feed_previous=self.testing)

        self.outputs, losses = seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs[:-1], targets,
            self.target_weights, self.bucket_sizes, seq2seq_function
        )
        self.train_ops = map(tf.train.AdadeltaOptimizer().minimize, losses)
        self.saver = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())

    def infer(self, articles, titles, sess):
        assert (articles.shape[0] == titles.shape[0])

        def slack(i, j):
            return self.bucket_sizes[i][j] - [articles, titles][j].shape[1]

        ids_of_buckets_that_fit = filter(lambda i: slack(i, 0) >= 0 and slack(i, 1) >= 0,
                                         range(len(self.bucket_sizes)))
        bucket_id = min(ids_of_buckets_that_fit,
                        key=lambda i: slack(i, 0) + slack(i, 1))

        encoder_size, decoder_size = self.bucket_sizes[bucket_id]
        target_weights = titles[:, 1:] != 0  # TODO
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
    seq_len1 = 1
    seq_len2 = 2
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

    # list of (article length, title length) pairs TODO
    bucket_sizes = [(article.shape[1], title.shape[1])
                    for (article, title) in
                    [(articles1, titles1), (articles2, titles2)]]
    bucket_sizes.sort(key=lambda size: size[0])

    with tf.Session() as session, tf.variable_scope('learn') as scope:

        rnn = Model(session, bucket_sizes, save_dir='main', testing=False,
                    go_code=1, depth=1, batch_size=batch_size,
                    embedding_dim=embedding_dim, hidden_size=hidden_size, memory_size=memory_size,
                    n_memory_slots=n_memory_slots, n_classes=n_classes)
        # rnn.load('main')
        # rnn.print_params()
        # output = rnn.infer(articles1, titles1, session)
        output = rnn.infer(articles2, titles2, session)
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

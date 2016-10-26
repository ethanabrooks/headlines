from pprint import pprint

from em_cell import ntmCell
from functools import partial

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell, LSTMCell, MultiRNNCell


class Model(object):
    """
    A sequence to sequence encoder-decoder using NTM cells
    """

    def __init__(self, session, bucket_sizes, save_dir, test_mode, depth, **kwargs):
        """
        :param session: tensorflow session
        :param bucket_sizes: list of (input length, output length) pairs
        :param save_dir: str: directory where variables are saved
        :param test_mode: bool: whether the model is used for testing
        (models used for testing feed output back into themselves during decoding)
        :param kwargs:
            go_code,
            depth,
            embedding_dim,
            hidden_size,
            n_memory_slots,
            n_classes
            TODO: these should be explicitly specified
        """
        self.session = session
        self.bucket_sizes = bucket_sizes
        self.__dict__.update(kwargs)

        # (max input length, max output length)
        max_sizes = (self.bucket_sizes[-1][0],
                     max(self.bucket_sizes, key=lambda size: size[1])[1])

        def make_placeholders(for_outputs, name, dtype=tf.int32):
            """
            :param for_outputs: bool: decoder inputs and target weights are for_outputs
            whereas encoder inputs are not.
            :param name: for placeholder variable
            :param dtype: for placeholder variable
            :return: list of n placeholders with unspecified shape,
            where n is the max size of either inputs or targets.
            """
            extension = for_outputs  # outputs are extended to accomodate <GO> symbol
            return [tf.placeholder(dtype, shape=[None], name=name + str(i))
                    for i in range(max_sizes[for_outputs] + extension)]

        # Feeds for inputs.
        self.encoder_inputs = make_placeholders(for_outputs=False, name='encoder')
        self.decoder_inputs = make_placeholders(for_outputs=True, name='decoder')
        self.target_weights = make_placeholders(for_outputs=True, name='weight',
                                                dtype=tf.float32)
        targets = self.decoder_inputs[1:]  # targets are decoder inputs shifted by one.

        # cell = MultiRNNCell([ntmCell(**kwargs) for _ in range(depth)])
        cell = ntmCell(**kwargs)

        # create actual model.
        # seq2seq.embedding_rnn_seq2seq embeds inputs
        # and runs them through a standard encoder-decoder model
        # but with our ntmCell
        def seq2seq_function(encoder_input, decoder_input):
            return seq2seq.embedding_rnn_seq2seq(
                encoder_input, decoder_input, cell,
                self.n_classes, self.n_classes, self.embedding_dim,
                feed_previous=test_mode)

        #
        # def seq2seq_function(encoder_input, decoder_input):
        #     return seq2seq.embedding_rnn_seq2seq(
        #         encoder_input, decoder_input, LSTMCell(kwargs['hidden_size']),
        #         self.n_classes, self.n_classes, self.embedding_dim,
        #         feed_previous=test_mode)

        # run the model.
        # seq2se2.model_with_buckets uses the buckets we defined earlier.
        self.outputs, self.losses = seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs[:-1], targets,
            self.target_weights, self.bucket_sizes, seq2seq_function
        )

        # minimize all losses
        self.train_ops = map(tf.train.AdadeltaOptimizer().minimize, self.losses)

        # saving/loading variables
        self.saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())

    def infer(self, intputs, outputs, sess):
        assert (intputs.shape[0] == outputs.shape[0])

        def slack(i, j):
            return self.bucket_sizes[i][j] - [intputs, outputs][j].shape[1]

        ids_of_buckets_that_fit = filter(lambda i: slack(i, 0) >= 0 and slack(i, 1) >= 0,
                                         range(len(self.bucket_sizes)))
        bucket_id = min(ids_of_buckets_that_fit,
                        key=lambda i: slack(i, 0) + slack(i, 1))

        encoder_size, decoder_size = self.bucket_sizes[bucket_id]
        target_weights = outputs[:, 1:] != 0  # TODO
        target_weights = np.c_[target_weights, np.zeros(outputs.shape[0])]

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for src, dest in [(intputs, self.encoder_inputs),
                          (outputs, self.decoder_inputs),
                          (target_weights, self.target_weights)]:
            for l in xrange(src.shape[1]):
                input_feed[dest[l].name] = src[:, l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([outputs.shape[0]], dtype=np.int32)

        pred_dist = tf.pack(self.outputs[bucket_id], axis=2)
        # adj_diffs = tf.sigmoid(pred_dist[:, :, 1:] - pred_dist[:, :, :-1])
        # repetition = tf.sigmoid(1 - (tf.reduce_sum(adj_diffs))
        output_feed = [tf.argmax(pred_dist, dimension=1),
                       self.losses[bucket_id],
                       self.train_ops[bucket_id]]
        outputs, loss, _ = sess.run(output_feed, input_feed)
        return outputs, loss

    def print_params(self):
        for var in tf.all_variables():
            print(var.name)
            print(self.session.run(var))


if __name__ == '__main__':
    dir = "train/5-3/"
    batch_size = 2
    seq_len1 = 8
    seq_len2 = 9
    hidden_size = 2
    embedding_dim = 5
    memory_dim = 7
    n_memory_slots = 2
    n_classes = batch_size * seq_len1 * hidden_size

    # [batch_size x seq_len] arrays
    articles1 = np.arange(batch_size * seq_len1, dtype='int32') \
        .reshape(batch_size, seq_len1)  # np.load(dir + "article.npy")
    titles1 = np.arange(batch_size * seq_len1, dtype='int32') \
        .reshape(batch_size, seq_len1)  # np.load(dir + "title.npy")
    articles2 = np.arange(batch_size * seq_len2, dtype='int32') \
        .reshape(batch_size, seq_len2)  # np.load(dir + "article.npy")
    titles2 = np.arange(batch_size * seq_len2, dtype='int32') \
        .reshape(batch_size, seq_len2)  # np.load(dir + "title.npy")

    # list of (article length, title length) pairs
    bucket_sizes = [(article.shape[1], title.shape[1])
                    for (article, title) in
                    [(articles1, titles1), (articles2, titles2)]]

    # sort bucket_sizes by article length
    bucket_sizes.sort(key=lambda lengths: lengths[0])

    with tf.Session() as session, tf.variable_scope('learn') as scope:

        rnn = Model(session, bucket_sizes, save_dir='main', test_mode=False, depth=1,
                    #  ntmCell params
                    go_code=1,
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size,
                    memory_dim=memory_dim,
                    n_memory_slots=n_memory_slots,
                    n_classes=n_classes)
        # rnn.load('main')
        # rnn.print_params()
        # output = rnn.infer(articles1, titles1, session)
        output = rnn.infer(articles2, titles2, session)


        def print_output(x):
            if type(x) == tuple or type(x) == list:
                print('-' * 10)
                for result in x:
                    print_output(result)
            else:
                print('-' * 10)
                print(x)
                try:
                    print(x.shape)
                except AttributeError:
                    pass


        print_output(output)

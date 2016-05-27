from __future__ import print_function
from __future__ import print_function

import argparse
import random
import subprocess
import sys
import time
import pickle
from collections import defaultdict
from collections import namedtuple
from functools import partial

import numpy as np
import os
from bokeh.io import output_file, vplot, save
from bokeh.plotting import figure
from rnn_em import Model
from tabulate import tabulate

# from spacy import English

parser = argparse.ArgumentParser()
parser.add_argument('--num_instances', type=int, default=100000,
                    help='number of instances to use in Jeopardy dataset')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding size')
parser.add_argument('--n_memory_slots', type=int, default=20, help='Memory slots')
parser.add_argument('--n_epochs', type=int, default=1000, help='Num epochs')
parser.add_argument('--seed', type=int, default=345, help='Seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of backprop through time steps')
parser.add_argument('--window_size', type=int, default=7,
                    help='Number of words in context window')
parser.add_argument('--learn_rate', type=float, default=0.0627142536696559,
                    help='Learning rate')
parser.add_argument('--verbose', help='Verbose or not', action='store_true')
parser.add_argument('--dataset', type=str, default='jeopardy',
                    help='select dataset [atis|Jeopardy]')
parser.add_argument('--plots', type=str, default='plots',
                    help='file for saving Bokeh plots output')
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')
parser.add_argument('--bucket_factor', type=int, default=2,
                    help='factor by which to multiply exponent when determining bucket size')

s = parser.parse_args()
print(s)
print('-' * 80)

""" Globals """
folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder):
    os.mkdir(folder)

np.random.seed(s.seed)
random.seed(s.seed)

PAD_VALUE = 0
NON_ANSWER_VALUE = 1
ANSWER_VALUE = 2
GO = '<go>'

assert s.window_size % 2 == 1, "`window_size` must be an odd number."


def get_bucket_idx(length):
    return int(np.math.ceil(np.math.log(length, s.bucket_factor)))



""" namedtuples """

Instance = namedtuple("instance", "article title")
Datasets = namedtuple("datasets", "train test")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")

""" classes """


class Dataset:
    def __init__(self):
        self.buckets = defaultdict(list)
        self.instances = Instance([], [])

    def fill_buckets(self):
        lengths = map(len, self.instances)
        assert lengths[0] == lengths[1]
        for article, title in zip(*self.instances):
            bucket_id = tuple(map(get_bucket_idx, [article.size, title.size]))
            self.buckets[bucket_id].append(Instance(article, title))


class Data:
    """
    contains global data parameters.
    Collects data and assigns to different datasets.
    """

    def __init__(self):

        self.sets = Datasets(Dataset(), Dataset())
        self.dictionary = Instance({GO: 0}, {GO: 0})
        self.vocsize = 0
        self.num_instances = 0
        self.num_train = 0

        def to_array(string, doc_type):
            if doc_type == 'article':
                string = GO + ' ' + string
            tokens = string.split()
            length = len(tokens)
            if not tokens:
                length += 1
            size = s.bucket_factor ** get_bucket_idx(length)
            sentence_vector = np.zeros(size, dtype='int32') + PAD_VALUE
            for i, word in enumerate(tokens):
                sentence_vector[i] = self.dictionary.__getattribute__(doc_type)[word]
            return sentence_vector

        for set_name in Datasets._fields:
            dataset = self.sets.__getattribute__(set_name)
            for doc_type in Instance._fields:

                def get_filename(extension):
                    return '.'.join([set_name, doc_type, extension])

                data_filename = get_filename('txt')
                if set_name == 'train':
                    dict_filename = get_filename('dict')
                    print(dict_filename)
                    with open(dict_filename) as dict_file:
                        for line in dict_file:
                            word, idx = line.split()
                            idx = int(float(idx))
                            self.vocsize = max(idx, self.vocsize)
                            self.dictionary.__getattribute__(doc_type)[word] = idx

                with open(os.path.join(s.data_dir, data_filename)) as data_file:
                    for line in data_file:
                        self.num_instances += 1
                        if set_name == 'train':
                            self.num_train += 1
                        array = to_array(line, doc_type)
                        dataset.instances.__getattribute__(doc_type).append(array)

            dataset.fill_buckets()
            print('Bucket allocation:')
            delete = []
            print('\nNumber of buckets: ', len(dataset.buckets))
            for key in dataset.buckets:
                bucket = dataset.buckets[key]
                num_instances = len(bucket)
                if num_instances < 10:
                    delete.append(key)
                else:
                    print(key, num_instances)

                dataset.buckets[key] = Instance(*map(np.array, zip(*bucket)))

            for key in delete:
                del (dataset.buckets[key])

            dataset.buckets = dataset.buckets.values()
            self.nclasses = len(self.dictionary.title)

    def print_data_stats(self):
        print("\nsize of dictionary:", self.vocsize)
        print("number of instances:", self.num_instances)
        print("size of training set:", self.num_train)


def get_batches(bucket):
    num_batches = bucket.article.shape[0] // s.batch_size + 1
    split = partial(np.array_split, indices_or_sections=num_batches)
    return zip(*map(split, (bucket.article,
                            bucket.title)))


def running_average(loss, new_loss, instances_processed, num_instances):
    if loss is None:
        return new_loss / instances_processed
    else:
        return (loss * (instances_processed - num_instances) + new_loss) / instances_processed


def print_progress(epoch, instances_processed, num_instances, loss, start_time):
    progress = round(float(instances_processed) / num_instances, ndigits=3)
    print('\r###\t{:<10d}{:<10.1%}{:<10.5f}{:<10.2f}###'
          .format(epoch, progress, float(loss), time.time() - start_time), end='')
    sys.stdout.flush()


def write_predictions_to_file(dataset_name, targets, predictions):
    filename = 'current.{0}.txt'.format(dataset_name)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as handle:
        for prediction_array, target_array in zip(predictions, targets):
            for prediction, target in zip(prediction_array, target_array):
                for label, arr in (('p: ', prediction), ('t: ', target)):
                    handle.write(label)
                    np.savetxt(handle, arr.reshape(1, -1), delimiter=' ', fmt='%i')
                    handle.write('\n')


def evaluate(predictions, targets):
    """
    @:param predictions: list of predictions
    @:param targets: list of targets
    @:return dictionary with entries 'f1'
    """

    def to_vector(list_of_arrays):
        return np.hstack(array.ravel() for array in list_of_arrays)

    predictions, targets = map(to_vector, (predictions, targets))

    metrics = np.zeros(3)

    def confusion((pred_is_pos, tgt_is_pos)):
        logical_and = np.logical_and(
            (predictions == ANSWER_VALUE) == pred_is_pos,
            (targets == ANSWER_VALUE) == tgt_is_pos
        )
        return logical_and.sum()

    tp, fp, fn = map(confusion, ((True, True), (True, False), (False, True)))
    metrics += np.array((tp, fp, fn))
    tp, fp, fn = metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return ConfusionMatrix(f1, precision, recall)


def print_random_scores(targets, predictions):
    predictions = [np.random.randint(low=NON_ANSWER_VALUE,
                                     high=ANSWER_VALUE + 1,
                                     size=array.shape) for array in predictions]
    confusion_matrix = evaluate(predictions, targets)
    print('\n' + tabulate(confusion_matrix.__dict__.iteritems(),
                          headers=["RANDOM", "score"]))


def track_scores(all_scores, confusion_matrix, epoch, dataset_name):
    Score = namedtuple("score", "value epoch")
    scores = all_scores[dataset_name]
    table = []
    for key in confusion_matrix._fields:
        result = confusion_matrix.__getattribute__(key)
        scores[key].append(Score(result, epoch))
        best_score = max(scores[key], key=lambda score: score.value)
        table.append([key, result, best_score.value, best_score.epoch])
        if result > best_score.value:
            command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dataset_name)
            subprocess.call(command.split())
    headers = [dataset_name.upper(), "score", "best score", "best score epoch"]
    print('\n\n' + tabulate(table, headers=headers))


def print_graphs(scores):
    output_file(s.plots + ".html")
    properties_per_dataset = {
        'train': {'line_color': 'firebrick'},
        'test': {'line_color': 'orange'},
        'valid': {'line_color': 'olive'}
    }

    plots = []
    for metric in ConfusionMatrix._fields:
        plot = figure(width=500, plot_height=500, title=metric)
        for dataset_name in scores:
            metric_scores = [score.value for score in scores[dataset_name][metric]]
            plot.line(range(len(metric_scores)),
                      metric_scores,
                      legend=dataset_name,
                      **properties_per_dataset[dataset_name])
        plots.append(plot)
    p = vplot(*plots)
    save(p)


if __name__ == '__main__':

    ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")

    data = Data()
    data.print_data_stats()

    rnn = Model(s.hidden_size,
                data.nclasses,
                data.vocsize,  # num_embeddings
                s.embedding_dim,  # embedding_dim
                1,  # window_size
                s.memory_size,
                s.n_memory_slots)

    scores = {dataset_name: defaultdict(list)
              for dataset_name in Datasets._fields}
    for epoch in range(s.n_epochs):
        print('\n###\t{:10}{:10}{:10}{:10}###'
              .format('epoch', 'progress', 'loss', 'runtime'))
        start_time = time.time()
        for name in list(Datasets._fields):
            random_predictions, predictions, targets = [], [], []
            instances_processed = 0
            loss = None
            for bucket in data.sets.__getattribute__(name).buckets:
                for articles, titles in get_batches(bucket):
                    if name == 'train':
                        bucket_predictions, new_loss = rnn.train(articles,
                                                                 titles)
                        rnn.normalize()
                        num_instances = articles.shape[0]
                        instances_processed += num_instances
                        loss = running_average(loss,
                                               new_loss,
                                               instances_processed,
                                               num_instances)
                        print_progress(epoch,
                                       instances_processed,
                                       data.num_train,
                                       loss,
                                       start_time)
                    else:
                        bucket_predictions = rnn.predict(articles, titles)

                    predictions.append(bucket_predictions.reshape(titles.shape))
                    targets.append(titles)
            write_predictions_to_file(name, predictions, targets)
            confusion_matrix = evaluate(predictions, targets)
            track_scores(scores, confusion_matrix, epoch, name)
            if name == 'test':
                print_random_scores(predictions, targets)
        print_graphs(scores)

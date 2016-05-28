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
parser.add_argument('--n_memory_slots', type=int, default=100, help='Memory slots')
parser.add_argument('--n_epochs', type=int, default=1000, help='Num epochs')
parser.add_argument('--seed', type=int, default=345, help='Seed')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Number of backprop through time steps')
parser.add_argument('--window_size', type=int, default=7,
                    help='Number of words in context window')
parser.add_argument('--learn_rate', type=float, default=0.0627142536696559,
                    help='Learning rate')
parser.add_argument('--verbose', help='Verbose or not', action='store_true')
parser.add_argument('--save_vars', help='pickle certain variables', action='store_true')
parser.add_argument('--load_vars', help='pickle.load certain variables', action='store_true')
parser.add_argument('--dataset', type=str, default='jeopardy',
                    help='select dataset [atis|Jeopardy]')
parser.add_argument('--plots', type=str, default='plots',
                    help='file for saving Bokeh plots output')
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')
parser.add_argument('--bucket_factor', type=int, default=4,
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
Score = namedtuple("score", "value epoch")

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
        self.to_int = Instance({}, {})
        self.to_word = Instance(defaultdict(list), defaultdict(list))
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
                sentence_vector[i] = self.to_int.__getattribute__(doc_type)[word]
            return sentence_vector

        for set_name in Datasets._fields:
            dataset = self.sets.__getattribute__(set_name)
            for doc_type in Instance._fields:

                def get_filename(extension):
                    return '.'.join([set_name, doc_type, extension])

                data_filename = get_filename('txt')
                if set_name == 'train':
                    dict_filename = get_filename('dict')
                    with open(dict_filename) as dict_file:
                        for line in dict_file:
                            word, idx = line.split()
                            idx = int(float(idx))
                            self.vocsize = max(idx, self.vocsize)
                            self.to_int.__getattribute__(doc_type)[word] = idx
                            self.to_word.__getattribute__(doc_type)[idx].append(word)

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
            self.nclasses = self.vocsize + 1

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


def write_predictions_to_file(to_word_dict, dataset_name, targets, predictions):

    def to_word(idx):
        word_list = to_word_dict[idx]
        if len(word_list) == 1:
            return word_list[0]
        if len(word_list) < 5:
            return '{' + '|'.join(word_list) + '}'
        else:
            return '<oov>'

    filename = 'current.{0}.txt'.format(dataset_name)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as handle:
        for prediction_array, target_array in zip(predictions, targets):
            for prediction, target in zip(prediction_array, target_array):
                for label, arr in (('p: ', prediction), ('t: ', target)):
                    values = ' '.join([to_word(idx) for idx in arr.ravel()])
                    handle.write(label + values + '\n')


def evaluate(predictions, targets):
    """
    @:param predictions: list of predictions
    @:param targets: list of targets
    @:return dictionary with entries 'f1'
    """

    def to_vector(list_of_arrays):
        return np.hstack(array.ravel() for array in list_of_arrays)

    predictions, targets = map(to_vector, (predictions, targets))
    return (predictions == targets).mean()


# def print_random_scores(targets, predictions):
#     predictions = [np.random.randint(low=NON_ANSWER_VALUE,
#                                      high=ANSWER_VALUE + 1,
#                                      size=array.shape) for array in predictions]
#     print('\n Random accuracy:', evaluate(predictions, targets))


def track_scores(scores, accuracy, epoch, dataset_name):
    scores[dataset_name].append(Score(accuracy, epoch))
    best_score = max(scores[dataset_name], key=lambda score: score.value)
    table = [['accuracy: ', accuracy, best_score.value, best_score.epoch]]
    if accuracy > best_score.value:
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
    plot = figure(width=500, plot_height=500, title='accuracy')
    for dataset_name in scores:
        accuracies = [score.value for score in scores[dataset_name]]
        plot.line(range(len(accuracies)),
                  accuracies,
                  legend=dataset_name,
                  **properties_per_dataset[dataset_name])
    plots.append(plot)
    p = vplot(*plots)
    save(p)


if __name__ == '__main__':

    data = Data()
    data.print_data_stats()

    if not s.load_vars:
        rnn = Model(s.hidden_size,
                    data.nclasses,
                    data.vocsize,  # num_embeddings
                    s.embedding_dim,  # embedding_dim
                    1,  # window_size
                    s.memory_size,
                    s.n_memory_slots)

    scores = {dataset_name: []
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

                    if s.save_vars:
                        with open('articles.pkl', 'w') as handle:
                            pickle.dump(articles, handle)
                        with open('titles.pkl', 'w') as handle:
                            pickle.dump(titles, handle)
                    if name == 'train':

                        if s.load_vars:
                            with open('bucket_predictions.pkl') as handle:
                                bucket_predictions = pickle.load(handle)
                            with open('new_loss.pkl') as handle:
                                new_loss = pickle.load(handle)
                        else:
                            bucket_predictions, new_loss = rnn.learn(articles, titles)
                            if s.save_vars:
                                with open('bucket_predictions.pkl', 'w') as handle:
                                    pickle.dump(bucket_predictions, handle)
                                with open('new_loss.pkl', 'w') as handle:
                                    pickle.dump(new_loss, handle)
                                exit(0)

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
                        if s.load_vars:
                            with open('bucket_predictions.pkl') as handle:
                                bucket_predictions = pickle.load(handle)
                        else:
                            bucket_predictions = rnn.infer(articles, titles)

                    print(bucket_predictions.shape, titles.shape)
                    try:
                        predictions.append(bucket_predictions.reshape(titles.shape))
                        targets.append(titles)
                    except ValueError:
                        pass
            write_predictions_to_file(data.to_word.title, name, predictions, targets)
            accuracy = evaluate(predictions, targets)
            track_scores(scores, accuracy, epoch, name)
            # if name == 'test':
            #     print_random_scores(predictions, targets)
        print_graphs(scores)
        exit(0)

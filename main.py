from __future__ import print_function
from __future__ import print_function

import argparse
import random
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from functools import partial
import numpy as np
import os
from bokeh.io import output_file, vplot, save
from bokeh.plotting import figure

from rnn_em import Model
from tabulate import tabulate
from pickle import dump, load

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding size')
parser.add_argument('--n_memory_slots', type=int, default=8, help='Memory slots')
parser.add_argument('--n_epochs', type=int, default=1000, help='Num epochs')
parser.add_argument('--seed', type=int, default=345, help='Seed')
parser.add_argument('--batch_size', type=int, default=20,
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

s = parser.parse_args()
assert s.window_size % 2 == 1, "`window_size` must be an odd number."
print(s)
print('-' * 80)

# from spacy import English

""" Globals """
folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder):
    os.mkdir(folder)


class Data:
    def __init__(self):
        pass


""" namedtuples """

Datasets = namedtuple("datasets", "train test")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")
Score = namedtuple("score", "value epoch")

""" functions """


def pickle(var_name, dir=''):
    with open(os.path.join(dir, var_name + '.pkl', 'w')) as handle:
        dump(eval(var_name), handle)


def unpickle(var_name, dir=''):
    with open(os.path.join(dir, var_name + '.pkl'), 'r') as handle:
        return load(handle)


def get_bucket_idx(length):
    return int(np.math.ceil(np.math.log(length, s.bucket_factor)))


def get_batches(bucket):
    num_batches = bucket[0].shape[0] // s.batch_size + 1
    split = partial(np.array_split, indices_or_sections=num_batches)
    return zip(*map(split, bucket))


def running_average(loss, new_loss, instances_processed, num_instances):
    if loss is None:
        return new_loss / instances_processed
    else:
        return (loss * (instances_processed - num_instances) + new_loss) / instances_processed


def print_progress(epoch, instances_processed, num_instances, loss, start_time, prediction):
    def format_time(seconds):
        if seconds is None:
            return float("nan")
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return ":".join((str(int(t)) for t in (hours, minutes, seconds)))

    def scientific_notation(x):
        if np.isnan(x):
            return 'NaN'
        exp = int(np.log10(x))
        sign = "+"
        if x < 1:
            sign = ""
            exp -= 1
        coeff = x * 10 ** (-exp)
        return "{:1.2}e{}{}".format(coeff, sign, exp)

    progress = round(float(instances_processed) / num_instances, ndigits=3)
    loss = scientific_notation(loss)
    elapsed_time = time.time() - start_time
    if instances_processed:
        eta = (num_instances - instances_processed) / instances_processed * elapsed_time
    else:
        eta = None
    elapsed_time, eta = map(format_time, (elapsed_time, eta))
    print('\r###\t{:<10d}{:<10.1%}{:<10}{:<10}{:<10}{}'
          .format(epoch, progress, loss, elapsed_time, eta, prediction[:100]), end='')
    sys.stdout.flush()


def translate(from_int, sep, pad, array):
    return sep.join((from_int[i] for i in array if from_int[i] != pad))


def write_predictions_to_file(from_int, pad, sep, arrays):
    assert type(from_int) == dict
    assert type(targets) == list
    assert type(predictions) == list

    def get_path(fname):
        return os.path.join(folder, fname)

    paths = map(get_path, ['targets.txt', 'predictions.txt'])
    for array_list, path in zip(arrays, paths):
        with open(path, 'w') as handle:
            for array in array_list:
                newlines = np.chararray((array.shape[0], 1))
                newlines[:] = '\n'
                vec_translate = np.vectorize(from_int.__getitem__)
                translated = vec_translate(array)
                string_array = np.c_[translated, newlines]
                if path == os.path.join(folder, 'targets.txt'):
                    string_array = string_array[:, 1:]  # remove <GO>
                remove_pads = string_array[np.where(string_array != pad)]
                string = sep.join(remove_pads.ravel()).replace(' \n ', '\n')
                handle.write(string)
    exit(0)  # TODO TODO TODO

            # for prediction_array, target_array in zip(predictions, targets):
            #     for prediction, target in zip(prediction_array, target_array):
            #
            #         for label, arr in (('p: ', prediction), ('t: ', target)):
            #             values = translate(arr.ravel(), from_int, sep, pad)
            #             handle.write(label + values + '\n')


def evaluate(predictions, targets):
    """
    @:param predictions: list of predictions
    @:param targets: list of targets
    @:return dictionary with entries 'f1'
    """

    def to_vector(list_of_arrays):
        return np.hstack(array.ravel() for array in list_of_arrays)

    targets = [target[:, 1:] for target in targets]  # get rid of <go> at beginning of targets
    predictions, targets = map(to_vector, (predictions, targets))
    assert predictions.size == targets.size
    return (predictions == targets).mean()


def track_scores(scores, accuracy, epoch, dataset_name):
    scores[dataset_name].append(Score(accuracy, epoch))
    best_score = max(scores[dataset_name], key=lambda score: score.value)
    table = [['accuracy: ', accuracy, best_score.value, best_score.epoch]]
    headers = [dataset_name.upper(), "score", "best score", "best score epoch"]
    print('\n\n' + tabulate(table, headers=headers))
    if accuracy >= best_score.value:
        command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dataset_name)
        subprocess.call(command.split())
        return True
    return False


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
    np.random.seed(s.seed)
    random.seed(s.seed)
    data = unpickle('data')
    Instance = namedtuple("instance", data.doc_types)

    print('loading model...')
    rnn = Model(s.hidden_size,
                data.nclasses,
                data.vocsize,  # num_embeddings
                s.embedding_dim,  # embedding_dim
                1,  # window_size
                s.memory_size,
                s.n_memory_slots,
                data.to_int[data.GO])

                # load_dir='main') TODO TODO TODO TODO
    rnn.print_params()

    scores = {dataset_name: []
              for dataset_name in Datasets._fields}
    for epoch in range(s.n_epochs):
        print('\n###\t{:10}{:10}{:10}{:10}{:10}{}'
              .format('epoch', 'progress', 'loss', 'runtime', 'ETA', 'sample prediction'))
        start_time = time.time()
        sample_prediction = None
        instances_processed = 0
        for set_name in list(Datasets._fields):
            # set_name = 'test'
            predictions, targets = [], []
            loss = None
            for bucket_dir in os.listdir(set_name):
                path = os.path.join(set_name, bucket_dir)
                filepaths = [os.path.join(path, name + '.npy')
                             for name in Instance._fields]
                instances = map(np.load, filepaths)
                assert instances[0].shape[0] == instances[1].shape[0]
                for articles, titles in get_batches(instances):
                    if set_name == 'train':
                        bucket_predictions, new_loss = rnn.learn(articles, titles)
                        num_instances = articles.shape[0]
                        instances_processed += num_instances
                        loss = running_average(loss,
                                               new_loss,
                                               instances_processed,
                                               num_instances)

                        if np.isinf(loss) or np.isnan(loss):
                            print('\nloss is ' + str(loss))
                            rnn.print_params()
                            exit(1)
                        if sample_prediction is None or time.time() - tic > 10:
                            tic = time.time()
                            print('')
                            sample_prediction = data.SEP.join([data.from_int[x] for x in bucket_predictions[0, :]])
                            rnn.save(folder)
                        print_progress(epoch,
                                       instances_processed,
                                       data.num_train,
                                       loss,
                                       start_time,
                                       sample_prediction)
                    else:
                        bucket_predictions = rnn.infer(articles, titles)
                    predictions.append(bucket_predictions)
                    targets.append(titles)

            accuracy = evaluate(predictions, targets)
            is_best_score = track_scores(scores, accuracy, epoch, set_name)
            if is_best_score:
                write_predictions_to_file(data.from_int, data.PAD, data.SEP, [targets, predictions])
            print_graphs(scores)

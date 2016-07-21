from __future__ import print_function

import argparse
import os
import pickle
import random

import numpy as np
from collections import defaultdict, namedtuple

import shutil

os.environ["THEANO_FLAGS"] = "device=cpu"

PAD = '<PAD>'
GO = '<GO>'
OOV = '<OOV>'
DATA_OBJ_FILE = 'data.pkl'


def get_bucket_idx(length):
    return int(np.math.ceil(np.math.log(length, s.bucket_factor)))


""" namedtuples """

doc_types = "comp simp"
Instance = namedtuple("instance", doc_types)
Datasets = namedtuple("datasets", "train test")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")
Score = namedtuple("score", "value epoch")

""" classes """


class Data:
    def __init__(self):
        self.PAD, self.GO, self.SEP = PAD, GO, ' '
        special_words = [PAD, GO, OOV]
        counts = {}
        for set_name in Instance._fields:
            dict_filename = 'train.article.dict'  # TODO: use a dict for Newsela
            dict_path = os.path.join(s.data_dir, dict_filename)
            with open(dict_path) as handle:
                for line in handle:
                    word, count = line.split()
                    counts[word] = float(count)

        self.to_int, self.from_int = dict(), dict()
        top_n_counts = sorted(counts, key=counts.__getitem__, reverse=True)[:s.size_vocab]
        for word in special_words + top_n_counts:
            idx = len(self.to_int)
            self.to_int[word] = idx
            self.from_int[idx] = word
        self.vocsize = len(self.to_int)
        self.nclasses = self.vocsize


""" functions """


def to_array(string, doc_type):
    tokens = string.split()
    if doc_type == 'simp':
        tokens = [GO] + tokens
    length = len(tokens)
    if not tokens:
        length += 1
    size = s.bucket_factor ** get_bucket_idx(length)
    sentence_vector = np.zeros(size, dtype='int32') + data.to_int[PAD]
    for i, word in enumerate(tokens):
        if word not in data.to_int:
            word = OOV
        sentence_vector[i] = data.to_int[word]
    return sentence_vector


def fill_buckets(instances):
    lengths = map(len, instances)
    assert lengths[0] == lengths[1]
    buckets = defaultdict(list)
    for article, title in zip(*instances):
        bucket_id = tuple(map(get_bucket_idx, [article.size, title.size]))
        buckets[bucket_id].append(Instance(article, title))
    return buckets


def save_buckets(num_train, buckets, set_name):
    print('\nNumber of buckets: ', len(buckets))
    for key in buckets:
        bucket = buckets[key]
        size_bucket = len(bucket)

        # we only keep buckets with more than 10 instances for optimization
        if size_bucket < 10 or bucket[0].comp.size == 0 or bucket[0].simp.size == 1:
            num_train -= size_bucket
        else:
            bucket_folder = os.path.join(set_name, '-'.join(map(str, key)))
            if not os.path.exists(bucket_folder):
                os.mkdir(bucket_folder)
            print(key, size_bucket)
            instance = Instance(*map(np.array, zip(*bucket)))
            for doc_type in Instance._fields:
                filepath = os.path.join(bucket_folder, doc_type)
                np.save(filepath, instance.__getattribute__(doc_type))


def print_stats(data):
    print("\nsize of dictionary:", data.vocsize)
    print("number of instances:", data.num_instances)
    print("size of training set:", data.num_train)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_instances', type=int, default=2000000000,
                        help='number of instances to use in Jeopardy dataset')
    parser.add_argument('--size_vocab', type=int, default=10000,
                        help='number of words in vocab')
    parser.add_argument('--data_dir', type=str, default='.',
                                                        # '/data2/jsedoc/fb_headline_first_sent/',
                        help='path to data')
    parser.add_argument('--bucket_factor', type=int, default=2,
                        help='factor by which to multiply exponent when determining bucket size')

    s = parser.parse_args()
    print(s)
    print('-' * 80)

    with open(DATA_OBJ_FILE, 'rb') as handle:
        data = pickle.load(handle)
    print('Bucket allocation:')
    for set_name in Datasets._fields:
        # start fresh every time
        if os.path.exists(set_name):
            shutil.rmtree(set_name)
        os.mkdir(set_name)

    instances = {'train': Instance([], []),
                 'test': Instance([], [])}
    seed = random.randint(0, 100)
    for doc_type in Instance._fields:
        data_filename = '.'.join(['tune.8turkers.tok', doc_type])
        random.seed(seed)  # ensures that the same sequence of random numbers is generated for simp and comp
        with open(os.path.join(s.data_dir, data_filename)) as data_file:
            for line in data_file:
                random_random = random.random()
                set_name = 'train' if random_random < .8 else 'test'
                array = to_array(line, doc_type)
                instance_arrays = instances[set_name].__getattribute__(doc_type)
                instance_arrays.append(array)
                if len(instance_arrays) == s.num_instances:
                    break

    for set_name in Datasets._fields:
        assert len(instances[set_name].comp) == len(instances[set_name].simp)
        buckets = fill_buckets(instances[set_name])
        save_buckets(len(instances['train'].comp), buckets, set_name)

    def num(set_name):
        return len(instances[set_name].comp)
    data.num_train, data.num_test = map(num, ['train', 'test'])
    data.num_instances = data.num_test + data.num_train
    data.doc_types = doc_types

    print_stats(data)
    with open(DATA_OBJ_FILE, 'w') as handle:
        pickle.dump(data, handle)

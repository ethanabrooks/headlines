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
        article_array, _ = article
        title_array, _ = title
        bucket_id = tuple(map(get_bucket_idx, [article_array.size, title_array.size]))
        buckets[bucket_id].append(Instance(article, title))
    return buckets


def save_buckets(num_train, buckets, set_name, data):
    print('\nNumber of buckets: ', len(buckets))
    for key in buckets:
        bucket = buckets[key]
        size_bucket = len(bucket)

        # we only keep buckets with more than 10 instances for optimization

        size_input = bucket[0].comp[0].size
        size_target = bucket[0].simp[0].size
        if size_bucket < 10 or size_input == 0 or size_target == 1:
            num_train -= size_bucket
        else:
            bucket_folder = os.path.join(set_name, '-'.join(map(str, key)))
            if not os.path.exists(bucket_folder):
                os.mkdir(bucket_folder)
            print(key, size_bucket)
            instances = zip(*bucket)
            for array_text_list, doc_type in zip(instances, doc_types.split()):
                filepath = os.path.join(bucket_folder, doc_type)
                arrays, text = zip(*array_text_list)
                np.save(filepath, np.array(arrays))
                with open(filepath + '.txt', 'w') as handle:
                    handle.write(''.join(text))
                # for array in arrays:
                #     print(filepath)
                #     print(' '.join([data.from_int[x] for x in array]))

def print_stats(data):
    print("\nsize of dictionary:", data.vocsize)
    print("number of instances:", data.num_instances)
    print("size of training set:", data.num_train)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_instances', type=int, default=20000000,
                        help='number of instances to use in Jeopardy dataset')
    parser.add_argument('--size_vocab', type=int, default=10000,
                        help='number of words in vocab')
    parser.add_argument('--data_dir', type=str,
                        default='.',
                        # default='/data2/jsedoc/Newsela'
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
                 'test':  Instance([], [])}

    seed = random.randint(0, 100)
    for set_name in Datasets._fields:
        for doc_type in Instance._fields:
            if set_name == 'test':
                data_path = 'tune.8turkers.tok.' + doc_type
            else:
                data_path = os.path.join(s.data_dir, 'newsela.train.' + doc_type + '.tok')
            random.seed(seed)  # ensures that the same sequence of random numbers is generated for simp and comp
            with open(data_path) as data_file:
                for line in data_file:
                    random_random = random.random()
                    array = to_array(line, doc_type)
                    instance_arrays = instances[set_name].__getattribute__(doc_type)
                    instance_arrays.append((array, line))
                    if len(instance_arrays) == s.num_instances:
                        break

    for set_name in Datasets._fields:
        assert len(instances[set_name].comp) == len(instances[set_name].simp)
        buckets = fill_buckets(instances[set_name])
        save_buckets(len(instances['train'].comp), buckets, set_name, data)

    def num(set_name):
        return len(instances[set_name].comp)
    data.num_train, data.num_test = map(num, ['train', 'test'])
    data.num_instances = data.num_test + data.num_train
    data.doc_types = doc_types

    print_stats(data)
    with open(DATA_OBJ_FILE, 'w') as handle:
        pickle.dump(data, handle)

import argparse
import string
import os
import pickle

import numpy as np
from collections import defaultdict, namedtuple
parser = argparse.ArgumentParser()
parser.add_argument('--num_instances', type=int, default=100000,
                    help='number of instances to use in Jeopardy dataset')
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')
parser.add_argument('--bucket_factor', type=int, default=4,
                    help='factor by which to multiply exponent when determining bucket size')

s = parser.parse_args()
print(s)
print('-' * 80)

from main import PAD, GO, OOV, DATA_OBJ_FILE


def get_bucket_idx(length):
    return int(np.math.ceil(np.math.log(length, s.bucket_factor)))


""" namedtuples """

Instance = namedtuple("instance", "article title")
Datasets = namedtuple("datasets", "train test")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")
Score = namedtuple("score", "value epoch")

""" classes """


class Data:
    def __init__(self):
        vocab = PAD + GO + OOV + '\n ' + string.lowercase + string.punctuation + string.digits
        self.to_char = dict(enumerate(vocab))
        self.to_int = {char: i for i, char in enumerate(vocab)}


""" functions """


def to_array(string, doc_type):
    length = len(string) + 1
    size = s.bucket_factor ** get_bucket_idx(length)
    if doc_type == 'title':
        string = GO + string
    sentence_vector = np.zeros(size, dtype='int32') + data.to_int[PAD]
    for i, char in enumerate(string):
        if char not in data.to_int:
            char = OOV
        char_code = data.to_int[char]
        sentence_vector[i] = char_code
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
    print('Bucket allocation:')
    print('\nNumber of buckets: ', len(buckets))
    for key in buckets:
        bucket = buckets[key]
        size_bucket = len(bucket)

        # we only keep buckets with more than 10 instances for optimization
        if size_bucket < 10:
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


if __name__ == '__main__':
    """
    contains global data parameters.
    Collects data and assigns to different datasets.
    """
    data = Data()
    data.num_train = 0
    for set_name in Datasets._fields:
        if not os.path.exists(set_name):
            os.mkdir(set_name)
        instances = Instance([], [])
        for doc_type in Instance._fields:
            num_instances = 0
            data_filename = '.'.join([set_name, doc_type, 'txt'])
            with open(os.path.join(s.data_dir, data_filename)) as data_file:
                for line in data_file:
                    num_instances += 1
                    if set_name == 'train':
                        data.num_train += 1
                    array = to_array(line, doc_type)
                    instances.__getattribute__(doc_type).append(array)
                    if num_instances == s.num_instances:
                        break

        buckets = fill_buckets(instances)
        save_buckets(data.num_train, buckets, set_name)
        data.nclasses = len(data.to_int)
        data.vocsize = data.nclasses
        with open(DATA_OBJ_FILE, 'w') as handle:
            pickle.dump(data, handle)

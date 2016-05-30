import argparse

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
        lengths = map(len, self.buckets)
        assert lengths[0] == lengths[1]
        new_buckets = defaultdict(list)
        for article, title in zip(*self.buckets):
            bucket_id = tuple(map(get_bucket_idx, [article.size, title.size]))
            new_buckets[bucket_id].append(Instance(article, title))
        self.buckets = new_buckets


class Data:
    """
    contains global data parameters.
    Collects data and assigns to different datasets.
    """

    def __init__(self):

        self.sets = Datasets(Dataset(), Dataset())
        self.num_instances = 0
        vocab = PAD + GO + OOV + '\n ' + string.lowercase + string.punctuation + string.digits
        self.to_char = dict(enumerate(vocab))
        self.to_int = {char: i for i, char in enumerate(vocab)}
        self.num_train = 0

        def to_array(string, doc_type):
            length = len(string) + 1
            size = s.bucket_factor ** get_bucket_idx(length)
            if doc_type == 'title':
                string = GO + string
            sentence_vector = np.zeros(size, dtype='int32') + self.to_int[PAD]
            for i, char in enumerate(string):
                if char not in self.to_int:
                    char = OOV
                char_code = self.to_int[char]
                sentence_vector[i] = char_code
            return sentence_vector

        for set_name in Datasets._fields:
            dataset = self.sets.__getattribute__(set_name)
            dataset.buckets = Instance([], [])
            for doc_type in Instance._fields:

                def get_filename(extension):
                    return '.'.join([set_name, doc_type, extension])

                data_filename = get_filename('txt')
                self.num_instances = 0
                with open(os.path.join(s.data_dir, data_filename)) as data_file:
                    for line in data_file:
                        self.num_instances += 1
                        if set_name == 'train' and doc_type == 'title':
                            self.num_train += 1
                        array = to_array(line, doc_type)
                        dataset.buckets.__getattribute__(doc_type).append(array)
                        if self.num_instances == s.num_instances:
                            break

            dataset.fill_buckets()
            print('Bucket allocation:')
            delete = []
            print('\nNumber of buckets: ', len(dataset.buckets))
            for key in dataset.buckets:
                bucket = dataset.buckets[key]
                num_instances = len(bucket)
                if num_instances < 10:
                    delete.append(key)
                    self.num_train -= len(bucket)
                else:
                    print(key, num_instances)

                dataset.buckets[key] = Instance(*map(np.array, zip(*bucket)))

            for key in delete:
                del (dataset.buckets[key])

            dataset.buckets = dataset.buckets.values()
            self.nclasses = len(self.to_int)
            self.vocsize = self.nclasses

    def print_data_stats(self):
        print("\nsize of dictionary:", self.vocsize)
        print("number of instances:", self.num_instances)
        print("size of training set:", self.num_train)

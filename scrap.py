import string

import numpy as np
import time
import os
from functools import partial
from pickle import dump, load
folder = 'main'

def unpickle(var_name, dir=''):
    with open(os.path.join(dir, var_name + '.pkl'), 'r') as handle:
        return load(handle)


def write_predictions_to_file(from_int, pad, sep, arrays):
    print(type(targets[0]))
    assert(type(targets[0]) == np.ndarray)
    def get_path(fname):
        return os.path.join(folder, fname)
    # filename = 'current.{0}.txt'.format(dataset_name)
    # filepath = os.path.join(folder, filename)
    paths = map(get_path, ['predictions', 'targets'])
    for array_list, path in zip(arrays, paths):
        for array in array_list:
            newlines = np.chararray((array.shape[0], 1))
            newlines[:] = '\n'
            vec_translate = np.vectorize(from_int.__getitem__)
            translated = vec_translate(array)
            string_array = np.c_[translated, newlines]
            remove_pads = string_array[np.where(string_array != pad)]
            string = sep.join(remove_pads.ravel()).replace(' \n ', '\n')
            print(string)

targets = [np.arange(i * 2).reshape(2, i) for i in range(1, 5)]
predictions = [np.arange(i * 2).reshape(2, i) for i in range(1, 5)]

from_int = {i: string.letters[i] for i in range(10)}
write_predictions_to_file(from_int, 'e', ' ', [targets, predictions])

import os
import csv
import numpy as np

doc_types = ['article', 'title']

for set_name in ['train', 'test']:
    for batch_dir in os.listdir(set_name):
        def get_path(doc_type):
            return os.path.join(set_name, batch_dir, doc_type + '.npy')


        filepaths = map(get_path, doc_types)
        arrays = map(np.load, filepaths)
        for array, path in zip(arrays, filepaths):
            path = path.replace('.npy', '.csv')
            array.tofile(path, sep=',', format='%10.5f')

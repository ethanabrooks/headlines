import argparse
import os
import pickle
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')

n = 40 * 1000
special_words = [['<pad>'], ['<go>'], ['<oov>']]
s = parser.parse_args()
counts = []
for set_name in ["article", "title"]:
    dictionary, reverse_dictionary = dict(), dict()
    dict_filename = 'train.' + set_name + '.dict.orig'
    print(dict_filename)
    dict_path = os.path.join(s.data_dir, dict_filename)
    print(dict_path)
    with open(dict_path) as handle:
        for line in handle:
            word, count = line.split()
            counts.append((word, float(count)))


top_n_counts = sorted(counts, key=itemgetter(1), reverse=True)[:n]
for word, _ in top_n_counts:
    dictionary[word] = len(dictionary)

dict_filename = 'dict.pkl'
with open(dict_filename, 'w+') as handle:
    pickle.dump(dictionary, handle)

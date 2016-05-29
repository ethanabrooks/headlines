import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')

special_words = [['<pad>'], ['<go>'], ['<oov>']]
s = parser.parse_args()
dictionary = dict()
reverse_dictionary = dict()
dict_filename = 'dict.txt'
dict_path = os.path.join(s.data_dir, dict_filename)
for set_name in ['test', 'train']:
    data_filename = 'dict.txt'
    data_path = os.path.join(s.data_dir, dict_filename)
    with open(dict_path) as handle:
        for line in handle:
            word, idx = line.split()
            dictionary[float(idx)].append(word)
            reverse_dictionary[word] = idx

    dict_filename = 'train.' + set_name + '.dict'
    with open(dict_filename, 'w+') as handle:
        for i, word_list in enumerate(special_words + dictionary.values()):
            for word in word_list:
                handle.write('{} {}\n'.format(word, i))
print(i)

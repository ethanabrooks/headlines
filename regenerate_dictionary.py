import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data2/jsedoc/fb_headline_first_sent/',
                    help='path to data')

dictionary = defaultdict(list)
special_words = [['<pad>'], ['<go>'], ['<oov>']]
s = parser.parse_args()
for set_name in ["article", "title"]:
    dict_filename = 'train.' + set_name + '.dict'
    print(dict_filename)
    dict_path = os.path.join(s.data_dir, dict_filename)
    print(dict_path)
    with open(dict_path) as handle:
        for line in handle:
            word, idx = line.split()
            dictionary[float(idx)].append(word)

    dict_filename = set_name + '.train.dict.new'
    with open(os.path.join(dict_filename), 'w+') as handle:
        for i, word_list in enumerate(special_words + dictionary.values()):
            for word in word_list:
                handle.write('{} {}\n'.format(word, i))
print(i)

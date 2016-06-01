from __future__ import print_function
import pickle


def print_stats(data):
    print("Training instances: ", data.num_train)
    print("Classes: ", data.nclasses)
    print("Vocabulary size: ", data.vocsize)

if __name__ == '__main__':
    class Data:
        def __init__(self): pass

    from main import DATA_OBJ_FILE

    with open(DATA_OBJ_FILE) as handle:
        data = pickle.load(handle)
    print_stats(data)

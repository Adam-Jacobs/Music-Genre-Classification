import numpy as np
import pickle
import sys


def load_numerical_data():
    print('Loading training & test data...')
    pickle_in = open("..\\feature_pickles\\train.pickle", "rb")
    train = np.array(pickle.load(pickle_in))

    pickle_in = open("..\\feature_pickles\\test.pickle", "rb")
    test = np.array(pickle.load(pickle_in))

    train_labels = [x.split(';') for x in train[:, 23]]
    test_labels = [x.split(';') for x in test[:, 23]]

    # Check that all songs have at least one label
    for labels in train_labels:
        if len(labels) == 0:
            print('An input song doesn\'t have an associated label')
            sys.exit()

    for labels in test_labels:
        if len(labels) == 0:
            print('An input song doesn\'t have an associated label')
            sys.exit()

    train_features = [[float(s) for s in x] for x in train[:, range(1, 23)]]
    test_features = [[float(s) for s in x] for x in test[:, range(1, 23)]]

    return train_features, train_labels, test_features, test_labels
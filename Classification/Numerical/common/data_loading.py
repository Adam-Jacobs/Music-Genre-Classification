import numpy as np
import pickle
import sys
sys.path.append("..\\..\\..\\common")
import data_manipulation as dm

def load_numerical_data(normalise=False):
    print('Loading training & test data...')
    pickle_in = open("..\\data\\train.pickle", "rb")
    train = np.array(pickle.load(pickle_in))

    pickle_in = open("..\\data\\test.pickle", "rb")
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

    if normalise:
        train_features = dm.normalise_features(train_features)
        test_features = dm.normalise_features(test_features)

    return train_features, train_labels, test_features, test_labels
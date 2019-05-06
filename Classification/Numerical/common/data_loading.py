import numpy as np
import pandas as pd
import pickle
import sys
import os
from common import data_manipulation as dm
from classification.numerical.common import setup_data


def load_numerical_data(features_file_path, normalise=False):
    print('Loading training & test data...')
    train_data_path = "classification\\numerical\\data\\train.csv"  # this needs to check for if this set up data comes
    test_data_path = "classification\\numerical\\data\\test.csv"  # from the same features_file_path currently specified

    if not os.path.isfile(train_data_path) or not os.path.isfile(test_data_path):
        print('Getting labels for train & test features specified...')
        setup_data.setup_data(features_file_path)

    train = pd.read_csv(train_data_path, dtype=None, delimiter=',', header=1, encoding='utf8')
    test = pd.read_csv(test_data_path, dtype=None, delimiter=',', header=1, encoding='utf8')

    train = np.array(train)
    test = np.array(test)

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
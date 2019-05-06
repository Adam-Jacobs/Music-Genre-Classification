import pickle
import numpy as np
import pandas as pd
import tqdm


def setup_data(features_file_path):
    numerical_unsorted = np.genfromtxt(features_file_path, dtype=None, delimiter=',', encoding='utf8')

    pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb") # change to data_loader
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb") # change to data_loader
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb") # change to data_loader
    test_labels = pickle.load(pickle_in)

    train_values = []
    test_values = []

    numerical_unsorted_ids = [x[0] for x in numerical_unsorted]

    print('Organising training set...')
    for track_id, labels in tqdm.tqdm(train_labels):
        if track_id in numerical_unsorted_ids:
            values = list(numerical_unsorted[numerical_unsorted_ids.index(track_id)])
            values.append(';'.join([str(x) for x in labels]))
            train_values.append(values)

    print('Organising testing set...')
    for track_id, labels in tqdm.tqdm(test_labels):
        if track_id in numerical_unsorted_ids:
            values = list(numerical_unsorted[numerical_unsorted_ids.index(track_id)])
            values.append(';'.join([str(x) for x in labels]))
            test_values.append(values)

    print('Saving data to csv...')
    df = pd.DataFrame(train_values)
    df.to_csv("feature_pickles\\train.csv", index=False, sep=',', encoding='utf-8')

    df = pd.DataFrame(test_values)
    df.to_csv("feature_pickles\\test.csv", index=False, sep=',', encoding='utf-8')

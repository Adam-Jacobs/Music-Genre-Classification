import pickle
import numpy as np
import tqdm


if __name__ == "__main__":
    features_unsorted = np.genfromtxt("..\\..\\Feature Extraction\\numerical features\\data\\features.csv",
                                      dtype=None, delimiter=',', encoding='utf8')

    pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    features_unsorted_ids = [x[0] for x in features_unsorted]

    train_features = []
    test_features = []

    print('Organising training set...')
    for track_id, labels in tqdm.tqdm(train_labels):
        train_features.append(features_unsorted[features_unsorted_ids.index('{:06d}'.format(int(track_id)))])

    print('Organising testing set...')
    for track_id, labels in tqdm.tqdm(test_labels):
        test_features.append(features_unsorted[features_unsorted_ids.index('{:06d}'.format(int(track_id)))])

    # Save the data
    print('Saving data to pickles...')
    pickle_out = open("feature_pickles\\train_features.pickle", "wb")
    pickle.dump(train_features, pickle_out)
    pickle_out.close()

    pickle_out = open("feature_pickles\\test_features.pickle", "wb")
    pickle.dump(test_features, pickle_out)
    pickle_out.close()

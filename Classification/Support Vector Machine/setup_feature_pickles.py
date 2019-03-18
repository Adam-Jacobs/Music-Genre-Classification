import pickle
import numpy as np
import tqdm


if __name__ == "__main__":
    numerical_unsorted = np.genfromtxt("..\\..\\Feature Extraction\\numerical features\\data\\features.csv",
                                      dtype=None, delimiter=',', encoding='utf8')

    pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    unsorted_ids = [x[0] for x in numerical_unsorted]
    unsorted_features = [x.tolist()[1:len(x)-1] for x in numerical_unsorted]

    train_features = []
    test_features = []

    print('Organising training set...')
    for track_id, labels in tqdm.tqdm(train_labels):
        if '{:06d}'.format(int(track_id)) in unsorted_ids:
            train_features.append(unsorted_features[unsorted_ids.index('{:06d}'.format(int(track_id)))])

    print('Organising testing set...')
    for track_id, labels in tqdm.tqdm(test_labels):
        if '{:06d}'.format(int(track_id)) in unsorted_ids:
            test_features.append(unsorted_features[unsorted_ids.index('{:06d}'.format(int(track_id)))])

    # Save the data
    print('Saving data to pickles...')
    pickle_out = open("feature_pickles\\train_features.pickle", "wb")
    pickle.dump(train_features, pickle_out)
    pickle_out.close()

    pickle_out = open("feature_pickles\\test_features.pickle", "wb")
    pickle.dump(test_features, pickle_out)
    pickle_out.close()

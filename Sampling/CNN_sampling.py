from imblearn.combine import SMOTEENN
import numpy as np
import pickle

if __name__ == "__main__":
    pickle_in = open("..\\Classification\\Convolutional Neural Network\\feature_pickles\\training_features.pickle", "rb")
    train_features = pickle.load(pickle_in)

    pickle_in = open("..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    # Reshape features for use with sampling
    train_features = train_features.reshape(-1, 101 * 240)
    train_labels = [x[1] for x in train_labels]

    samp = SMOTEENN()
    samp_features, samp_labels = samp.fit_resample(train_features, train_labels)

    print('Old data count: ' + str(len(train_labels)))
    print('New data count: ' + str(len(samp_labels)))

    # Save sampled data
    print('Saving sampled data to pickles...')
    pickle_out = open("data\\samp_train_features.pickle", "wb")
    pickle.dump(samp_features, pickle_out)
    pickle_out.close()

    pickle_out = open("data\\samp_train_labels.pickle", "wb")
    pickle.dump(samp_labels, pickle_out)
    pickle_out.close()

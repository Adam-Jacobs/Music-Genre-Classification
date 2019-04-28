import pickle


def get_genre_names():
    pickle_in = open("dataset labels\\pickles\\genre_names.pickle", "rb")
    return pickle.load(pickle_in)


def get_training_labels():
    pickle_in = open("dataset labels\pickles\\training_labels.pickle", "rb")
    return pickle.load(pickle_in)


def get_validation_labels():
    pickle_in = open("dataset labels\pickles\\validation_labels.pickle", "rb")
    return pickle.load(pickle_in)


def get_test_labels():
    pickle_in = open("dataset labels\pickles\\training_labels.pickle", "rb")
    return pickle.load(pickle_in)

import numpy as np
import pylab
import tsne
import pickle


genre_names = []
pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\genre_names.pickle", "rb")
genre_names.extend(pickle.load(pickle_in))


def get_genre_name(id):
    for index, list in enumerate(genre_names):
        if str(list[0]) == str(id):
            return list[1]
    return 'biiitch'


def get_labels():
    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\training_labels.pickle", "rb")
    training_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\validation_labels.pickle", "rb")
    validation_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    # Combine into one list for data visualisation
    label_lists = []
    label_lists.extend(training_labels)
    #label_lists.extend(validation_labels)
    #label_lists.extend(test_labels)

    # Get only the top level ID
    label_ids = [x[1][0] for x in label_lists] # This needs to be changed into logic to actually look for the top level genre

    # Get the genre name from the ID to display
    labels = [get_genre_name(id) for id in label_ids]

    return labels


if __name__ == "__main__":
    X = np.loadtxt("features_noid.csv", dtype='float', delimiter=',')
    Y = tsne.tsne(X, 2, 50, 100.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, get_labels()[:200])
    pylab.show()

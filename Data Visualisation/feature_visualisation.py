import numpy as np
import matplotlib.pyplot as plt
import tsne
import pickle

# TODO Clear untagged tracks

genre_names = []
pickle_in = open("..\\dataset labels\pickles\\genre_names.pickle", "rb")
genre_names.extend(pickle.load(pickle_in))

genre_top_levels = []
pickle_in = open("..\\dataset labels\pickles\\genre_top_levels.pickle", "rb")
genre_top_levels.extend(pickle.load(pickle_in))

top_level_available_colours = ["red", "yellow", "orange", "darkred",
                                   "blue", "purple", "darkblue", "cyan",
                                   "green", "darkgreen", "lawngreen", "black",
                                   "slategrey", "magenta", "teal", "olive"]


def get_genre_name(id):
    for _, pair in enumerate(genre_names):
        if str(id) == str(pair[0]):
            return pair[1]
    return 'Error'


def get_genre_top_level(id):
    for _, pair in enumerate(genre_top_levels):
        if str(id) == str(pair[0]):
            return pair[1]
    return -1


def get_genre_colours(top_level_ids):
    colours = []
    used_genres = []
    colour_counter = 0
    for _, genre in enumerate(top_level_ids):
        if genre not in [i[0] for i in used_genres]:
            if str(genre) == '-1':
                used_genres.append([genre, 'white'])
            else:
                used_genres.append([genre, top_level_available_colours[colour_counter]])
                colour_counter += 1

        colours.append(used_genres[[i[0] for i in used_genres].index(genre)][1])

    return colours


def get_labels_with_ignored_duplicates(labels):
    new_labels = []
    for _, label in enumerate(labels):
        if label not in new_labels:
            new_labels.append(label)
        else:
            new_labels.append('_' + label)

    return new_labels


def get_labels():
    pickle_in = open("..\\dataset labels\pickles\\training_labels.pickle", "rb")
    training_labels = pickle.load(pickle_in)

    pickle_in = open("..\\dataset labels\pickles\\validation_labels.pickle", "rb")
    validation_labels = pickle.load(pickle_in)

    pickle_in = open("..\\dataset labels\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    # Combine into one list for data visualisation
    label_lists = []
    label_lists.extend(training_labels)
    #label_lists.extend(validation_labels)
    #label_lists.extend(test_labels)

    # Get only the top level ID
    label_ids = [x[1][0] for x in label_lists]
    labels_top_level = [get_genre_top_level(id) for id in label_ids]
    colours = get_genre_colours(labels_top_level)
    labels = [get_genre_name(id) for id in labels_top_level]

    labels = get_labels_with_ignored_duplicates(labels);

    return colours, labels


if __name__ == "__main__":
    X = np.loadtxt("features.csv", dtype='float', delimiter=',', usecols=list(range(1, 22)))
    Y = tsne.tsne(X, 2, 50, 100.0)
    colours, labels = get_labels()

    # temp while data for all tracks hasn't been created yet
    colours = colours[:200]
    labels = labels[:200]

    fig, ax = plt.subplots()
    for i in range(0, len(labels)-1):
        ax.scatter(Y[:, 0][i], Y[:, 1][i], 20, c=colours[i], label=labels[i])

    plt.legend(loc='lower left', ncol=3, fontsize=8)
    plt.show()

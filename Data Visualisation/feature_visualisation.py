import numpy as np
import matplotlib.pyplot as plt
import tsne
import pickle
import sys
sys.path.append("..\\common")
import label_manipulation as lm

genre_names = []
pickle_in = open("..\\dataset labels\pickles\\genre_names.pickle", "rb")
genre_names.extend(pickle.load(pickle_in))

top_level_available_colours = ["red", "yellow", "orange", "darkred",
                                   "blue", "purple", "darkblue", "cyan",
                                   "green", "darkgreen", "lawngreen", "black",
                                   "slategrey", "magenta", "teal", "olive"]


def get_genre_name(id):
    for _, pair in enumerate(genre_names):
        if str(id) == str(pair[0]):
            return pair[1]
    return 'Error'


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


def get_labels_with_ignored_duplicates(tracks):  # This is for the plot legend
    new_labels = []
    for _, track in enumerate(tracks):
        if track.genre not in new_labels:
            new_labels.append(track.genre)
        else:
            track.genre = '_' + track.genre


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
    label_lists.extend(validation_labels)
    label_lists.extend(test_labels)

    # Get only the top level ID
    genre_ids = [x[1][0] for x in label_lists]
    labels_top_level = [lm.get_genre_top_level(id) for id in genre_ids]
    colours = get_genre_colours(labels_top_level)
    genre_labels = [get_genre_name(id) for id in labels_top_level]
    
    track_ids = [x[0] for x in label_lists]

    return [track_ids, colours, genre_labels]


class Track:

    def __init__(self, id, features, colour, genre):
        self.id = id
        self.features = features
        self.colour = colour
        self.genre = genre


def create_tracks(ids, features, plot_labels):
    tracks = []

    for id_index, id in enumerate(ids):
        # Check if track has tagged genre data
        if id in plot_labels[0]:
            labels_index = plot_labels[0].index(id)
            tracks.append(Track(id, features[id_index], plot_labels[1][labels_index], plot_labels[2][labels_index]))

    return tracks


def create_tsne_plot(perplexity, num_tracks, image_name):
    print('Setting up data...')
    track_ids = np.loadtxt("..\\Feature Extraction\\numerical features\\data\\features.csv", dtype='int',
                           delimiter=',', usecols=0)
    track_features = np.loadtxt("..\\Feature Extraction\\numerical features\\data\\features.csv", dtype=None,
                                delimiter=',', usecols=range(1, 22))

    track_plot_labels = get_labels()

    # Structure data to be more readable in Track objects
    tracks = create_tracks(track_ids, track_features, track_plot_labels)[:num_tracks]

    # Remove any duplicate labels for the legend to be accurate
    get_labels_with_ignored_duplicates(tracks)

    ax_vals = tsne.tsne(np.array([track.features for track in tracks]), 2, 50, perplexity)

    fig, ax = plt.subplots()
    for i in range(0, len(tracks) - 1):
        ax.scatter(ax_vals[:, 0][i], ax_vals[:, 1][i], 20, c=tracks[i].colour, label=tracks[i].genre)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=4, fontsize=10)
    plt.savefig('plots\\' + image_name + '.png', bbox_inches="tight")

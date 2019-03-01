import numpy as np
import pandas as pd
import os
import random
import pickle
import sys
sys.path.append("..\\common")
import label_manipulation as lm


def load_metadata():
    genre_names = np.genfromtxt("genres.csv",
                                dtype=None, delimiter=',', skip_header=1, usecols=(0, 3), encoding='utf8')
    genre_top_levels = np.genfromtxt("genres.csv",
                                     dtype=None, delimiter=',', skip_header=1, usecols=(0, 4), encoding='utf8')
    track_genres = pd.read_csv("tracks_genres_cleaned.csv",
                               quotechar='"', skipinitialspace=True, header=0, usecols=['track_id', 'genres'])

    return genre_names, genre_top_levels, track_genres.values


training_metadata = []
validation_metadata = []
test_metadata = []


def populate_data_subsets(data):
    # Put the data in a random order to evenly distribute throughout the subsets
    random.shuffle(data)
    # Put the first 80% into training set
    training_metadata.extend(data[:int(len(data) * 0.8)])
    # Put the next 10% into validation set
    validation_metadata.extend(data[len(training_metadata):len(training_metadata) + int(len(data) * 0.1)])
    # Put the remaining data (final 10%) into test set
    test_metadata.extend(data[len(training_metadata) + len(validation_metadata):])


def get_labels_from_genre_tags(genre_tags):
    # Change the format of the genre ids for this track from a string '[a, b, c]' to an array of genre ids
    return str(genre_tags).replace('[', '').replace(']', '').replace('\"', '').replace(' ', '').split(',')


def filter_untagged_tracks(unfiltered_list):
    track_genres = []
    for _, item in enumerate(unfiltered_list):
        genres = get_labels_from_genre_tags(item[1])
        if len(genres) > 0 and genres[0] != '':
            track_genres.append([item[0], lm.get_genre_top_level(genres[0])])
    return track_genres


def save_genre_pickles(genre_names, genre_top_levels):
    pickle_out = open("pickles\\genre_names.pickle", "wb")
    pickle.dump(genre_names, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\genre_top_levels.pickle", "wb")
    pickle.dump(genre_top_levels, pickle_out)
    pickle_out.close()


def setup_data():
    metadata = load_metadata()
    save_genre_pickles(metadata[0], metadata[1])

    filtered_data = filter_untagged_tracks(metadata[2])
    populate_data_subsets(filtered_data)


if __name__ == "__main__":
    print('Loading data...')
    setup_data()

    # Save the data
    print('Saving data to pickles...')
    pickle_out = open("pickles\\training_labels.pickle", "wb")
    pickle.dump(training_metadata, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\validation_labels.pickle", "wb")
    pickle.dump(validation_metadata, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\testing_labels.pickle", "wb")
    pickle.dump(test_metadata, pickle_out)
    pickle_out.close()

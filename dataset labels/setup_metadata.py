import numpy as np
import os
import random
import pickle


def load_metadata():
    genres = np.genfromtxt("genres.csv",
                           dtype=None, delimiter=',', skip_header=1, usecols=(0, 3), encoding='utf8')
    track_genres = np.genfromtxt("tracks_genres_cleaned.csv",
                                 dtype=None, delimiter=',', skip_header=1, usecols=(0, 2), encoding='utf8')
    return genres, track_genres


def filter_untagged_tracks(unfiltered_list):
    filtered_list = [x for x in unfiltered_list if '[]' not in x[1]]
    return filtered_list


def populate_data_subsets(data):
    # Put the data in a random order to evenly distribute throughout the subsets
    random.shuffle(data)
    # Put the first 80% into training set
    training_metadata.extend(data[:int(len(data) * 0.8)])
    # Put the next 10% into validation set
    validation_metadata.extend(data[len(training_metadata):len(training_metadata) + int(len(data) * 0.1)])
    # Put the remaining data (final 10%) into test set
    test_metadata.extend(data[len(training_metadata) + len(validation_metadata):])


genre_names = []
training_metadata = []
validation_metadata = []
test_metadata = []


def setup_data():
    metadata = load_metadata()
    genre_names.extend(metadata[0])
    populate_data_subsets(filter_untagged_tracks(metadata[1]))
    del metadata


def get_labels_from_genre_tags(genre_tags):
    # Change the format of the genre ids for this track from a string '[a, b, c]' to an array of genre ids
    return genre_tags.replace('[', '').replace(']', '').replace('\"', '').replace(' ', '').split(',')


if __name__ == "__main__":
    print('Loading data...')
    setup_data()

    print('Sorting data...')
    training_labels = []
    validation_labels = []
    test_labels = []

    for track_id, genres in training_metadata:
        training_labels.append([track_id, get_labels_from_genre_tags(genres)])

    for track_id, genres in validation_metadata:
        validation_labels.append([track_id, get_labels_from_genre_tags(genres)])

    for track_id, genres in test_metadata:
        test_labels.append([track_id, get_labels_from_genre_tags(genres)])

    # Save the data
    print('Saving data to pickles...')
    pickle_out = open("pickles\\genre_names.pickle", "wb")
    pickle.dump(genre_names, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\training_labels.pickle", "wb")
    pickle.dump(training_labels, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\validation_labels.pickle", "wb")
    pickle.dump(validation_labels, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles\\testing_labels.pickle", "wb")
    pickle.dump(test_labels, pickle_out)
    pickle_out.close()

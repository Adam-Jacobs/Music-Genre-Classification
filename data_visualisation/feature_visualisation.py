import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import sys
from data_visualisation import tsne
from common.label_manipulation import LabelManipulator
from common.data_manipulation import DataManipulation as dm
import data_loader

genre_names = data_loader.get_genre_names()


top_level_colours = {'Rock': 'red',
                     'International': 'yellow',
                     'Blues': 'orange',
                     'Jazz': 'darkred',
                     'Classical': 'blue',
                     'Old-Time / Historic': 'purple',
                     'Country': 'darkblue',
                     'Pop': 'cyan',
                     'Easy Listening': 'green',
                     'Soul-RnB': 'darkgreen',
                     'Electronic': 'lawngreen',
                     'Folk': 'black',
                     'Spoken': 'slategrey',
                     'Hip-Hop': 'magenta',
                     'Experimental': 'teal',
                     'Instrumental': 'olive'}


class Track:

    def __init__(self):
        self.id = -1
        self.features = []
        self.colour = ''
        self.genre_id = -1
        self.genre_name = ''


def get_genre_name(genre_id):
    for pair in genre_names:
        if str(genre_id) == str(pair[0]):
            return pair[1]
    return 'Error'


'''Takes list of Track objects with at least id and genre_id properties populated
Returns list of Track objects, additionally with the colour populated'''
def get_genre_colours(tracks):
    for track in tracks:
        track.genre_name = get_genre_name(track.genre_id)
        track.colour = top_level_colours[track.genre_name]

    return tracks


def get_labels_with_ignored_duplicates(tracks):  # For the plot legend
    new_labels = []
    for track in tracks:
        if track.genre_id not in new_labels:
            new_labels.append(track.genre_id)
        else:
            track.genre_name = '_' + str(track.genre_name)


def get_plottable_tracks():
    track_labels = []
    track_labels.extend(data_loader.get_training_labels())
    track_labels.extend(data_loader.get_validation_labels())
    track_labels.extend(data_loader.get_test_labels())
    # i.e - [track_id, track_genres[]]

    # Filter any tracks out that have more than one top-level genre
    viable_track_labels = []

    lm = LabelManipulator()

    for info in track_labels:
        top_level_genre_ids = []
        # Get all top_level genres the track is labelled with
        for genre_id in info[1]:
            top_level_genre_ids.append(lm.get_genre_top_level(lm.uncategorise_genre(genre_id)))

        # Remove any duplicates
        top_level_genre_ids = list(set(top_level_genre_ids))
        if len(top_level_genre_ids) == 1:
            track = Track()
            track.id = info[0]
            track.genre_id = top_level_genre_ids[0]
            viable_track_labels.append(track)

    viable_track_labels = get_genre_colours(viable_track_labels)

    return viable_track_labels


def copy(old_track):
    new_track = Track()
    new_track.id = old_track.id
    new_track.genre_id = old_track.genre_id
    new_track.genre_name = old_track.genre_name
    new_track.colour = old_track.colour
    new_track.features = old_track.features
    return new_track


def populate_track_features(ids, features, original_tracks):
    new_tracks = []

    print('creating id list...')
    feature_ids = [int(x) for x in ids]

    print('looping through tracks')
    for track in tqdm.tqdm(original_tracks):
        if track.id in feature_ids:  # This step is only required while not all features have been extracted
            new_track = copy(track)
            new_track.features = features[feature_ids.index(track.id)]
            new_tracks.append(new_track)

    return new_tracks


'''Saves a scatterplot .png image to save path specified'''
def create_tsne_plot(perplexity, num_tracks, features_file_path, image_name, normalise=False, save_path=None):
    print('Setting up data...')

    track_features_and_ids = np.loadtxt(features_file_path, dtype=None, delimiter=',')
    track_ids = [x[0] for x in track_features_and_ids]
    track_features = [x[1:len(x)] for x in track_features_and_ids]

    # Normalise features if requested
    if normalise:
        track_features = dm.normalise_features(track_features)

    tracks = get_plottable_tracks()[:num_tracks]

    print('populating track features...')
    # Populate track objects with features
    tracks = populate_track_features(track_ids, track_features, tracks)

    # Remove any duplicate labels for the legend to be accurate
    get_labels_with_ignored_duplicates(tracks)

    ax_vals = tsne.tsne(np.array([track.features for track in tracks]), 2, 50, perplexity)

    fig, ax = plt.subplots()
    for i in range(0, len(tracks) - 1):
        ax.scatter(ax_vals[:, 0][i], ax_vals[:, 1][i], 20, c=tracks[i].colour, label=tracks[i].genre_name)

    # Place legend aesthetically
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=4, fontsize=10)

    if save_path is None:
        # Construct directoy path to save image to
        dir_path = "plots_" + feature_type + '\\' + "single-label\\"
        if not normalise:
            dir_path += "not "
        dir_path += "normalised\\"
    else:
        dir_path = save_path

    # Save image
    plt.savefig(dir_path + '\\' + image_name + '.png', bbox_inches="tight")

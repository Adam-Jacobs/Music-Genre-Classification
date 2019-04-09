import numpy as np


'''Takes an array of floats
Returns a normalised list of floats'''
def min_max_normalise(features):
    # Min-Max Equation: xnorm = (x - min) / (max - min)

    minVal = min(features)
    maxVal = max(features)

    new_features = []
    for x in features:
        new_features.append((x - minVal)/(maxVal - minVal))

    return new_features


'''Takes matrix of floats where [i] is all features for a track and [i][j] is a float value
returns matrix in same format with normalised values for each respective feature dimension'''
def normalise_features(track_features):
    track_features = np.array(track_features)

    # Change shape to where [i] is equal to a list of values of one feature for all tracks
    features_reshaped = []
    for i in range(0, len(track_features[0])):
        features_reshaped.append(track_features[:, i])

    # Normalise all values according to their respective feature
    normalised_values_reshaped = []
    for single_feature_value_list in features_reshaped:
        normalised_values_reshaped.append(min_max_normalise(single_feature_value_list))

    # Change shape back to original shape where [i] is all features for a track
    normalised_track_features = []
    for i in range(0, len(normalised_values_reshaped[0])):
        normalised_single_track_features = []
        for j in range(0, len(normalised_values_reshaped)):
            normalised_single_track_features.append(normalised_values_reshaped[j][i])

        normalised_track_features.append(normalised_single_track_features)

    return normalised_track_features

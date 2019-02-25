import numpy as np
import os
import cv2
import pickle
import tqdm
from multiprocessing import Pool


# Dimensions of spectrogram
image_startX = 81
image_endX = 803
image_startY = 36
image_endY = 341
image_sizeX = image_endX - image_startX
image_sizeY = image_endY - image_startY
image_size = image_sizeX * image_sizeY


def load_spectrograms():
    spectrogramsPath = "..\\..\\..\\..\\..\\..\\FYP_Data\\spectrogram_images"
    data = []
    spectrogram_file_paths = os.listdir(spectrogramsPath)
    for img in tqdm.tqdm(spectrogram_file_paths):
        try:
            # Read the image into memory
            data_point = cv2.imread(os.path.join(spectrogramsPath, img), cv2.IMREAD_GRAYSCALE)

            # Crop to include only the spectrogram (thus removing the axis and labelling)
            data_point = data_point[image_startY:image_endY, image_startX:image_endX]

            # Scale the image down for better memory management
            data_point = cv2.resize(data_point, (0, 0), fx=0.3, fy=0.3)  # ideally temporary until I figure a better way

            # Append the features and the track number to the list
            data.append([data_point, img.split('.')[0]])
        except Exception as e:
            print("Exception for reading image", e, os.path.join(spectrogramsPath, img))

    return data


spectrograms = []
training_labels = []
validation_labels = []
test_labels = []


def setup_data():
    print('Loading spectrograms...')
    spectrograms.extend(load_spectrograms())

    print('Loading subset labels...')
    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\training_labels.pickle", "rb")
    training_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\validation_labels.pickle", "rb")
    validation_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\testing_labels.pickle", "rb")
    test_labels.extend(pickle.load(pickle_in))


def find_spectrogram_features(track_id):
    for index, item in enumerate(spectrograms):
        if item[1] == track_id:
            return item[0]


if __name__ == "__main__":
    print('Setting up data...')
    setup_data()

    training_features = []
    validation_features = []
    test_features = []

    print('Organising features into corect subsets...')
    for track_id, labels in training_labels:
        training_features.append(find_spectrogram_features(str(track_id)))

    for track_id, labels in validation_labels:
        validation_features.append(find_spectrogram_features(str(track_id)))

    for track_id, labels in test_labels:
        test_features.append(find_spectrogram_features(str(track_id)))

    # Reshape the features ready for use with CNN
    # training_features = np.array(training_features).reshape(-1, image_sizeX, image_sizeY, 1)
    # validation_features = np.array(validation_features).reshape(-1, image_sizeX, image_sizeY, 1)
    # test_features = np.array(test_features).reshape(-1, image_sizeX, image_sizeY, 1)

    # Save the data
    print('Saving data to pickles...')

    pickle_out = open("feature_pickles\\training_features.pickle", "wb")
    pickle.dump(training_features, pickle_out)
    pickle_out.close()

    pickle_out = open("feature_pickles\\validation_features.pickle", "wb")
    pickle.dump(validation_features, pickle_out)
    pickle_out.close()

    pickle_out = open("feature_pickles\\test_features.pickle", "wb")
    pickle.dump(test_features, pickle_out)
    pickle_out.close()

    # csv
    allfeatures = []
    allfeatures.extend(training_features)
    allfeatures.extend(validation_features)
    allfeatures.extend(test_features)

    np.savetxt("data\\features.csv", allfeatures, delimiter=",", fmt="%s")

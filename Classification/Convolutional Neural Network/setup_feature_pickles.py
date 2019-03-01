import numpy as np
import os
import os.path
import cv2
import pickle
import tqdm
from multiprocessing import Pool

# TODO - find better way than down-scaling the images

# Dimensions of spectrogram
image_startX = 81
image_endX = 803
image_startY = 36
image_endY = 341
image_sizeX = image_endX - image_startX
image_sizeY = image_endY - image_startY
image_size = image_sizeX * image_sizeY
image_scaled_sizeX = int(image_sizeX / 3)
image_scaled_sizeY = int(image_sizeY / 3)
spectrograms_features_path = "feature_pickles\\spectrogram_features_unsorted.pickle"


def load_spectrogram(path):
    try:
        # Read the image into memory
        data_point = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Crop to include only the spectrogram (thus removing the axis and labelling)
        data_point = data_point[image_startY:image_endY, image_startX:image_endX]

        # Scale the image down for better memory management
        data_point = cv2.resize(data_point, (image_scaled_sizeX, image_scaled_sizeY))

        return [os.path.basename(path).split('.')[0], data_point]
    except Exception as e:
        print("Exception for reading image", e, path)

    return None


def load_spectrograms():
    spectrograms_dir = "..\\..\\..\\..\\..\\..\\FYP_Data\\spectrogram_images"
    spectrogram_file_paths = os.listdir(spectrograms_dir)
    spectrogram_file_paths = [os.path.join(spectrograms_dir, x) for x in spectrogram_file_paths]
    spectrogram_file_paths = spectrogram_file_paths

    data = []

    with Pool(os.cpu_count() - 1) as pool:
        data.extend(list(tqdm.tqdm(pool.imap(load_spectrogram, spectrogram_file_paths),
                                   total=len(spectrogram_file_paths))))

    return data


spectrograms_ids = []
spectrograms_features = []
training_labels = []
test_labels = []


def setup_data():
    print('Loading spectrograms...')

    # If this process has been run before, continue from the completion of loading in the spectrograms
    if os.path.isfile(spectrograms_features_path):
        pickle_in = open(spectrograms_features_path, "rb")
        spectrograms = pickle.load(pickle_in)
    else:
        spectrograms = load_spectrograms()
        pickle_out = open(spectrograms_features_path, "wb")
        pickle.dump(spectrograms, pickle_out)
        pickle_out.close()

    spectrograms_ids.extend([x[0] for x in spectrograms])
    spectrograms_features.extend([x[1] for x in spectrograms])

    print('Loading subset labels...')
    pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    training_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    training_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
    test_labels.extend(pickle.load(pickle_in))


if __name__ == "__main__":
    print('Setting up data...')
    setup_data()

    print('Organising features into correct subsets...')
    pool = Pool(os.cpu_count() - 1)

    training_features = []
    test_features = []

    print('Organising training set...')
    for track_id, labels in tqdm.tqdm(training_labels):
        training_features.append(spectrograms_features[spectrograms_ids.index('{:06d}'.format(int(track_id)))])

    print('Organising testing set...')
    for track_id, labels in tqdm.tqdm(test_labels):
        test_features.append(spectrograms_features[spectrograms_ids.index('{:06d}'.format(int(track_id)))])

    # csv
    # allfeatures = []
    # allfeatures.extend(training_features)
    # allfeatures.extend(test_features)

    # np.savetxt("feature_pickles\\features.csv", [x[1] if x is not None else None for x in allfeatures],
    #            delimiter=",", fmt="%s")

    # Reshape the features ready for use with CNN
    training_features = np.array(training_features).reshape(-1, image_scaled_sizeY, image_scaled_sizeX, 1)
    test_features = np.array(test_features).reshape(-1, image_scaled_sizeY, image_scaled_sizeX, 1)

    # Save the data
    print('Saving data to pickles...')
    pickle_out = open("feature_pickles\\training_features.pickle", "wb")
    pickle.dump(training_features, pickle_out)
    pickle_out.close()

    pickle_out = open("feature_pickles\\test_features.pickle", "wb")
    pickle.dump(test_features, pickle_out)
    pickle_out.close()

import numpy as np
import pickle

# Dimensions of spectrogram
image_startX = 81
image_endX = 803
image_startY = 36
image_endY = 341
image_sizeX = image_endX - image_startX
image_sizeY = image_endY - image_startY
image_size = image_sizeX * image_sizeY

pickle_in = open("feature_pickles\\training_features.pickle", "rb")
training_features = pickle.load(pickle_in)

pickle_in = open("feature_pickles\\validation_features.pickle", "rb")
validation_features = pickle.load(pickle_in)

pickle_in = open("feature_pickles\\test_features.pickle", "rb")
test_features = pickle.load(pickle_in)

print('len(training_features[0]):' + str(len(training_features[0])))
print('training_features[0]')
print(training_features[0])
print('len(training_features[0][0]):' + str(len(training_features[0][0])))
print('training_features[0][0]')
print(training_features[0][0])
print('training_features[1][4]')
print(training_features[1][4])

print('RESHAPING')
#(int(image_sizeX / 3), int(image_sizeY / 3))
training_features = np.array(training_features).reshape(-1, int(image_sizeX / 3), int(image_sizeY / 3), 1)

print('len(training_features[0]):' + str(len(training_features[0])))
print('training_features[0]')
print(training_features[0])
print('len(training_features[0][0]):' + str(len(training_features[0][0])))
print('training_features[0][0]')
print(training_features[0][0])
print('training_features[1][4]')
print(training_features[1][4])

    # validation_features = np.array(validation_features).reshape(-1, image_sizeX, image_sizeY, 1)
    # test_features = np.array(test_features).reshape(-1, image_sizeX, image_sizeY, 1)

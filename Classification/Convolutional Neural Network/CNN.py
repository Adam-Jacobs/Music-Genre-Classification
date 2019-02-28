import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

#pickle_in = open("genres.pickle", "rb")
#genres = pickle.load(pickle_in)

pickle_in = open("feature_pickles\\training_features.pickle", "rb")
training_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
training_labels = pickle.load(pickle_in)

# X = X/255.0 # is normalisation needed?

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=training_features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature maps to 1D vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)

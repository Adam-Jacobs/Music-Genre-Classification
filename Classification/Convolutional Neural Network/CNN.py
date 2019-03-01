import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

import pickle

pickle_in = open("feature_pickles\\training_features.pickle", "rb")
train_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
train_labels = pickle.load(pickle_in)

pickle_in = open("feature_pickles\\test_features.pickle", "rb")
test_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
test_labels = pickle.load(pickle_in)

#temp to see how it trains
train_features = train_features[:1000]
train_labels = train_labels[:1000]

# train_features = train_features/255.0 # is normalisation needed?
model = Sequential()

# Input layer
model.add(Conv2D(256, (3, 3), input_shape=train_features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature maps to 1D vectors

model.add(Dense(64))

# Output layer
model.add(Dense(16))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_features, to_categorical(np.array([x[1] for x in train_labels])),
          batch_size=16, epochs=2, validation_split=((len(train_labels) + len(test_labels))/10))

model.summary()

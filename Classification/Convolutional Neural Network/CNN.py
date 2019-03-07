import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import model_state_IO as modelIO
import pickle
import os


os.system('mode con: cols=180 lines=40')
model_name = input('Please input the name of this model: ')

print('Loading training & test data...')
pickle_in = open("feature_pickles\\training_features.pickle", "rb")
train_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
train_labels = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
train_labels.extend(pickle.load(pickle_in))

pickle_in = open("feature_pickles\\test_features.pickle", "rb")
test_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
test_labels = pickle.load(pickle_in)

# train_features = train_features/255.0 # is normalisation needed?

print('Creating CNN model...')
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

print('Training CNN...')
model.fit(train_features, to_categorical(np.array([x[1] for x in train_labels])),
          batch_size=12, epochs=3,
          validation_split=(((len(train_labels) + len(test_labels))/10) / (len(train_labels) + len(test_labels))))

print('Evaluating CNN performance...')
scores = model.evaluate(test_features, to_categorical(np.array([x[1] for x in test_labels])))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

modelIO.save_model(model, model_name)

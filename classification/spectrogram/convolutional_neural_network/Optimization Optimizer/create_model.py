import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
# import model_state_IO as modelIO
import os
import pickle
import model_attributes as MA
import csv
import sys
sys.path.append("..\\")
import model_state_IO as modelIO

os.system('mode con: cols=220 lines=40')


def get_shape_from_size(size):
    shape = (size, size)
    return shape


def train_model(model_attributes):
    print('Loading training & test data...')
    pickle_in = open("..\\feature_pickles\\training_features.pickle", "rb")
    train_features = pickle.load(pickle_in)

    pickle_in = open("..\\..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\feature_pickles\\test_features.pickle", "rb")
    test_features = pickle.load(pickle_in)

    pickle_in = open("..\\..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    print('Creating CNN model...')
    model = Sequential()

    model.add(Conv2D(model_attributes.layers[i].num_neurons, get_shape_from_size(model_attributes.layers[i].conv_size),
                     input_shape=train_features.shape[1:]))
    if model_attributes.layers[i].activation_name != '':
        model.add(Activation(model_attributes.layers[i].activation_name))

    for i in range(1, len(model_attributes.layers) - 1):
        layer = model_attributes.layers[i]
        if layer.type_name == 'Conv2D':
            model.add(Conv2D(layer.num_neurons, get_shape_from_size(layer.conv_size)))
        elif layer.type_name == 'MaxPooling2D':
            model.add(MaxPooling2D(layer.num_neurons, get_shape_from_size(layer.conv_size)))
        elif layer.type_name == 'Dense':
            model.add(Dense(layer.num_neurons))

        if layer.activation_name != '':
            model.add(Activation(layer.activation_name))

    # Output setup
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('sigmoid'))

    if model_attributes.optimizer_name == 'SGD':
        optimizer = optimizers.SGD(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'Adagrad':
        optimizer = optimizers.Adagrad(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'Adadelta':
        optimizer = optimizers.Adadelta(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'Adam':
        optimizer = optimizers.Adam(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'Adamax':
        optimizer = optimizers.Adamax(lr=get_learning_rate())
    elif model_attributes.optimizer_name == 'Nadam':
        optimizer = optimizers.Nadam(lr=get_learning_rate())

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    print('Training CNN...')
    model.fit(train_features, to_categorical(np.array([x[1] for x in train_labels])),
              batch_size=model_attributes.batch_size, epochs=model_attributes.num_epochs,
              validation_split=model_attributes.validation_split)

    print('Evaluating CNN performance...')
    scores = model.evaluate(test_features, to_categorical(np.array([x[1] for x in test_labels])))

    return scores

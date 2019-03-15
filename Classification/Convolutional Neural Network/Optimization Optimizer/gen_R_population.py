import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
# import model_state_IO as modelIO
import os
import pickle
import CNN_R_model_creation as Create
import model_attributes as MA
import csv
import sys
import time
sys.path.append("..\\")
import model_state_IO as modelIO

os.system('mode con: cols=220 lines=40')
# model_name = input('Please input the name of this model: ')

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

# train_features = train_features/255.0 # is normalisation needed?

for _ in range(1000):
    Create.reset()
    print('Creating CNN model...')
    model = Sequential()

    layers = Create.get_layers(train_features.shape[1:])

    for layer in layers:
        try:
            model.add(layer)  # Can sometimes mess up with MaxPooling layer if the input from previous layer isn't right
        except:
            continue

    model.add(Flatten())

    # Output layer
    model.add(Dense(16))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Create.get_optimizer(),
                  metrics=['categorical_accuracy'])

    print('Training CNN...')

    start_train_time = time.time()
    model.fit(train_features, to_categorical(np.array([x[1] for x in train_labels])),
              batch_size=Create.get_batch_size(), epochs=Create.get_num_epochs(),
              validation_split=Create.get_validation_split_num())
    end_train_time = time.time()

    print('Evaluating CNN performance...')
    scores = model.evaluate(test_features, to_categorical(np.array([x[1] for x in test_labels])))

    Create.model_attributes.loss = scores[0]
    Create.model_attributes.accuracy = scores[1]
    Create.train_time = end_train_time - start_train_time

    filename = "model_configs.csv"

    # Open configs file
    write_header = False
    if os.path.isfile(filename):
        models_csv = open(filename, "r+")
    else:
        models_csv = open(filename, "w+")
        write_header = True

    csv_writer = csv.writer(models_csv, delimiter=',', lineterminator='\n')

    if write_header:
        csv_writer.writerow(Create.model_attributes.get_header_writable())

    # Set id for current config
    Create.model_attributes.id = sum(1 for line in models_csv)

    # Save this config
    csv_writer.writerow(Create.model_attributes.get_writable())

    models_csv.close()

    # Save model
    modelIO.save_model(model, str(Create.model_attributes.id))

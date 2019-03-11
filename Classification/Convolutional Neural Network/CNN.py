import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import MultiLabelBinarizer
import model_state_IO as modelIO
import pickle
import os
import sys
sys.path.append("..\\..\\common")
import label_manipulation as lm


os.system('mode con: cols=180 lines=40')
#model_name = input('Please input the name of this model: ')


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


def clear_genres_list(genres, accepted):
    new_genres = []
    for genre in genres:
        if genre in accepted:
            new_genres.append(genre)

    return new_genres


def cut_genres_from_list(features, labels):
    new_features = []
    new_labels = []
    accepted_genres = [lm.categorise_genre(12), lm.categorise_genre(15), lm.categorise_genre(38)]

    for i in range(0, len(labels)):
        if lm.categorise_genre(12) in labels[i][1] or lm.categorise_genre(15) in labels[i][1] or lm.categorise_genre(38) in labels[i][1]:
            new_features.append(features[i])
            new_labels.append([labels[i][0], clear_genres_list(labels[i][1], accepted_genres)])

    return np.array(new_features), np.array(new_labels)


# Test accuracy with only best populated genres -> Experimental (38k), Electronic (34k), Rock (33k)
def cut_all_but_3_genres():
    global train_features, train_labels
    train_features, train_labels = cut_genres_from_list(train_features, train_labels)

    global test_features, test_labels
    test_features, test_labels = cut_genres_from_list(test_features, test_labels)


# cut_all_but_3_genres()

prediction_features = test_features[2001: 2010]
prediction_labels = test_labels[2001: 2010]

train_features = train_features[:2000]
train_labels = train_labels[:2000]
test_features = test_features[:2000]
test_labels = test_labels[:2000]

print('Creating CNN model...')
model = Sequential()

# Input layer
model.add(Conv2D(256, (3, 3), input_shape=train_features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer
#model.add(Conv2D(256, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature maps to 1D vectors

model.add(Dense(64))

# Output layer
model.add(Dense(16))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

print('Training CNN...')
mlb = MultiLabelBinarizer()
model.fit(train_features, mlb.fit_transform(np.array([x[1] for x in train_labels])),
          batch_size=12, epochs=1,
          validation_split=(((len(train_labels) + len(test_labels))/10) / (len(train_labels) + len(test_labels))))

print('Evaluating CNN performance...')
scores = model.evaluate(test_features, mlb.fit_transform(np.array([x[1] for x in test_labels])))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(np.array(prediction_features))

print('Predictions: ')
print(predictions)

categorised_prediction_labels = mlb.fit_transform(np.array([x[1] for x in prediction_labels]))
labels_to_output = []
for i, l in enumerate(prediction_labels):
    labels_to_output.append([l[0], categorised_prediction_labels[i]])

print('Correct labels:')
print(labels_to_output)

# modelIO.save_model(model, model_name)

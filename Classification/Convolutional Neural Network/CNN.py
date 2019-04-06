import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from sklearn.preprocessing import MultiLabelBinarizer
import model_state_IO as modelIO
import pickle
import os
import gc
import sys
sys.path.append("..\\..\\common")
import label_manipulation as lm


# os.system('mode con: cols=180 lines=40')
# model_name = input('Please input the name of this model: ')


print('Loading training & test data...')
pickle_in = open("feature_pickles\\downscaled3\\training_features.pickle", "rb")
train_features = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
train_labels = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
train_labels.extend(pickle.load(pickle_in))

pickle_in = open("feature_pickles\\downscaled3\\test_features.pickle", "rb")
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


def normalise_data():
    global train_features, test_features

    for x in range(0, len(train_features)):
        for y in range(0, len(train_features[x])):
            train_features[x][y] = train_features[x][y] / 255

    for x in range(0, len(test_features)):
        for y in range(0, len(test_features[x])):
            test_features[x][y] = test_features[x][y] / 255


def data_to_int(data):
    for x in range(0, len(data)):
        for y in range(0, len(data[x])):
            data[x][y] = int(data[x][y])

    return data


# cut_all_but_3_genres()
# normalise_data()

train_features = train_features[:2000]
train_labels = train_labels[:2000]
test_features = test_features[:2000]
test_labels = test_labels[:2000]

pred_point = len(test_features) - 8

prediction_features = test_features[pred_point:]
prediction_labels = test_labels[pred_point:]

test_features = test_features[:pred_point]
test_labels = test_labels[:pred_point]

print('Creating CNN model...')
model = Sequential()

# Input layer
model.add(Conv2D(757, (3, 3), activation='relu', input_shape=train_features.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(757, (3, 3), activation='relu'))
#model.add(Conv2D(256, (20, 20), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature maps to 1D vectors

# model.add(Dropout(rate=0.2))
# model.add(Dense(64))
model.add(Dense(64))

# Output layer
model.add(Dense(16))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['categorical_accuracy'])

print('Training CNN...')
mlb = MultiLabelBinarizer()
model.fit(train_features, mlb.fit_transform(np.array([x[1] for x in train_labels])),
          batch_size=12, epochs=1,
          validation_split=(((len(train_labels) + len(test_labels))/10) / (len(train_labels) + len(test_labels))))

print('Evaluating CNN performance...')
scores = model.evaluate(test_features, mlb.fit_transform(np.array([x[1] for x in test_labels])))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

prediction_features1 = prediction_features[:4]
prediction_labels1 = prediction_labels[:4]
prediction_features2 = prediction_features[4:]
prediction_labels2 = prediction_labels[4:]

predictions1 = model.predict(np.array(prediction_features1))
predictions2 = model.predict(np.array(prediction_features2))

predictions1 = data_to_int(predictions1)
predictions2 = data_to_int(predictions2)

print('Predictions: ')
print(predictions1)
print(predictions2)

# prediction_labels.append(['', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
categorised_prediction_labels1 = mlb.fit_transform(np.array([x[1] for x in prediction_labels1]))
categorised_prediction_labels2 = mlb.fit_transform(np.array([x[1] for x in prediction_labels2]))

# track_id + one-hot labels
# labels_to_output = []
# for i, l in enumerate(prediction_labels):
#     labels_to_output.append([l[0], categorised_prediction_labels[i]])

print('Correct labels:')
print(categorised_prediction_labels1)
print(categorised_prediction_labels2)

# modelIO.save_model(model, model_name)

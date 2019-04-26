from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import sys
import pickle
sys.path.append("..\\common")
import data_loading
sys.path.append("..\\..\\..\\common")
import label_manipulation as lm


def clear_genres_list(genres, accepted):
    new_genres = []
    for genre in genres:
        if genre in accepted:
            new_genres.append(genre)

    return new_genres


def cut_genres_from_list(features, labels):
    new_features = []
    new_labels = []
    accepted_genres = [str(lm.categorise_genre(12)), str(lm.categorise_genre(15)), str(lm.categorise_genre(38))]

    for i in range(0, len(labels)):
        if str(lm.categorise_genre(12)) in labels[i] or str(lm.categorise_genre(15)) in labels[i] or str(lm.categorise_genre(38)) in labels[i]:
            new_features.append(features[i])
            new_labels.append(clear_genres_list(labels[i], accepted_genres))

    return np.array(new_features), np.array(new_labels)


# Test accuracy with only best populated genres -> Experimental (38k), Electronic (34k), Rock (33k)
def cut_all_but_3_genres(train_features, train_labels, test_features, test_labels):
    train_features, train_labels = cut_genres_from_list(train_features, train_labels)
    test_features, test_labels = cut_genres_from_list(test_features, test_labels)

    return train_features, train_labels, test_features, test_labels

train_features, train_labels, test_features, test_labels = data_loading.load_numerical_data(normalise=False)

# train_features, train_labels, test_features, test_labels = cut_all_but_3_genres(train_features, train_labels, test_features, test_labels)

MultiLabelBinarizer.set_params(range(0, 16))
mlb = MultiLabelBinarizer()
train_labels = np.array(train_labels)
np.random.shuffle(train_labels)
train_labels = mlb.fit_transform(np.array(train_labels))
test_labels = mlb.fit_transform(np.array(test_labels))

predict_features = test_features[:4]
predict_labels = test_labels[:4]

test_features = test_features[4:]
test_labels = test_labels[4:]

print('Training RF...')
model = RandomForestClassifier(verbose=1)
model.fit(train_features, train_labels)

print('Evaluating RF...')
print('Accuracy: ' + str(model.score(test_features, test_labels)))

print()
print('Model Predictions:')
print(np.array([[int(s) for s in x] for x in model.predict(predict_features)]))

print()
print('Correct Labels:')
print(predict_labels)

pickle.dump(model, open("model\\model.pickle", 'wb'))

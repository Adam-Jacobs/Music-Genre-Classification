from sklearn import svm
import numpy as np
import pickle

train_features = []
test_features = []
train_labels = []
test_labels = []


def load_labels():
    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\train_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("..\\..\\fma_metadata_applied\\pickles\\test_labels.pickle", "rb")
    test_labels.extend(pickle.load(pickle_in))


def setup_features():
    all_features = np.genfromtxt("..\\..\\fma_metadata_original\\features.csv",
                                 delimiter=',', skip_header=4, usecols=(0, 2), encoding='utf8')
    while len(train_features) < len(train_labels) or len(test_features) < len(test_labels):
        for index, track_features in enumerate(all_features):
            if track_features[0] == train_labels[len(train_features)][0]:
                train_features.append(np.delete(track_features, 0))
            elif len(test_features) < len(test_labels):
                if track_features[0] == test_labels[len(test_features)][0]:
                    test_features.append(np.delete(track_features, 0))


print('Loading Labels...')
load_labels()

print('Setting Up Features...')
setup_features()

print('Length of Training Features: ' + str(len(train_features)))
print('Length of Testing Features: ' + str(len(test_features)))
print('Length of Training Labels: ' + str(len(train_labels)))
print('Length of Testing Labels: ' + str(len(test_labels)))

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
print('Training SVM...')
clf.fit(train_features, train_labels)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print('Evaluating SVM...')
print('Accuracy: ' + str(clf.accuracy(test_features, test_labels)))

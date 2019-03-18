from sklearn import svm
import numpy as np
import pickle


def load_data():
    print('Loading training & test data...')
    pickle_in = open("feature_pickles\\train_features.pickle", "rb")
    train_features = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
    train_labels = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
    train_labels.extend(pickle.load(pickle_in))

    pickle_in = open("feature_pickles\\test_features.pickle", "rb")
    test_features = pickle.load(pickle_in)

    pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
    test_labels = pickle.load(pickle_in)

    return train_features, train_labels, test_features, test_labels


train_features, train_labels, test_features, test_labels = load_data()

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
print('Training SVM...')
clf.fit(train_features, train_labels)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print('Evaluating SVM...')
print('Accuracy: ' + str(clf.accuracy(test_features, test_labels)))

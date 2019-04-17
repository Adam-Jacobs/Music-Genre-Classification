from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import sys
sys.path.append("..\\common")
import data_loading


train_features, train_labels, test_features, test_labels = data_loading.load_numerical_data()

print('Training SVM...')
mlb = MultiLabelBinarizer()

clf = OneVsRestClassifier(estimator=LinearSVC(multi_class='ovr', verbose=1))
clf.fit(train_features, mlb.fit_transform(np.array(train_labels)))

print('Evaluating SVM...')
print('Accuracy: ' + str(clf.score(test_features, mlb.fit_transform(np.array(test_labels)))))

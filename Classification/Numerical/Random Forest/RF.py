from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import sys
sys.path.append("..\\common")
import data_loading


train_features, train_labels, test_features, test_labels = data_loading.load_numerical_data()

mlb = MultiLabelBinarizer()
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

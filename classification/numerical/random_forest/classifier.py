from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.ensemble
import pickle
import os
import numpy as np
from classification.common.classifier_template import ClassifierTemplate
from common.label_manipulation import LabelManipulator
from classification.numerical.common import data_loading


class RandomForestClassifier(ClassifierTemplate):
    def __init__(self):
        pass

    def classify(self, features):
        model = pickle.load(open("classification\\numerical\\random_forest\\model\\model.pickle", 'rb'))
        result = model.predict(features)

        # todo put this functionality into the common classifier template
        MultiLabelBinarizer.set_params(range(0, 16))
        mlb = MultiLabelBinarizer()
        mlb.fit([range(0, 16)])
        genre_predictions_categorized = mlb.inverse_transform(result)

        if len(genre_predictions_categorized) == 0 or not all(genre_predictions_categorized):
            return ["Unclassifiable"]

        genre_predictions_categorized = [x[0] for x in mlb.inverse_transform(result)]  # this needs checkinf for which value o fthe tuple is the actual value

        genre_predictions = []
        lm = LabelManipulator()
        for label in genre_predictions_categorized:
            genre_predictions.append(lm.uncategorise_genre(label))

        # convert the ids to names

        return genre_predictions

    def train(self, features_file_path, save_dir_path):
        train_features, train_labels, test_features, test_labels = data_loading.load_numerical_data(features_file_path, normalise=False)

        MultiLabelBinarizer.set_params(range(0, 16))
        mlb = MultiLabelBinarizer()
        train_labels = np.array(train_labels)
        # Used to create a baseline for random chance
        # np.random.shuffle(train_labels)
        train_labels = mlb.fit_transform(np.array(train_labels))
        test_labels = mlb.fit_transform(np.array(test_labels))

        # Reserve the first 4 tracks in test set for displaying predictions to dev
        predict_features = test_features[:4]
        predict_labels = test_labels[:4]

        test_features = test_features[4:]
        test_labels = test_labels[4:]

        print('Training RF...')
        model = sklearn.ensemble.RandomForestClassifier(verbose=1)
        model.fit(train_features, train_labels)

        print('Evaluating RF...')
        print('Accuracy: ' + str(model.score(test_features, test_labels)))

        print()
        print('Model Predictions:')
        print(np.array([[int(s) for s in x] for x in model.predict(predict_features)]))

        print()
        print('Correct Labels:')
        print(predict_labels)

        pickle.dump(model, open(os.path.join(save_dir_path, "model.pickle"), 'wb'))

from feature_extraction.numerical import numerical_feature_extraction as numerical_FE
from classification.numerical.random_forest.classifier import RandomForestClassifier
from common.label_manipulation import LabelManipulator


def classify(mp3_path):
    features = numerical_FE.extract_features(mp3_path)
    classifier = RandomForestClassifier()
    prediction = classifier.classify([features])

    if prediction[0] == 'Unclassifiable':
        return 'Other'

    genre_names = []
    lm = LabelManipulator()
    for id in prediction:
        genre_names.append(lm.get_genre_name(id))

    return ''.join(genre_names)


def train(classifier_name, features_file_path, save_dir_path):
    if classifier_name == 'Random Forest':
        classifier = RandomForestClassifier()
    elif classifier_name == 'Convolutional Neural Network':
        pass  # Add logic for 2nd classifier

    classifier.train(features_file_path, save_dir_path)

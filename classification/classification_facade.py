from feature_extraction.numerical import numerical_feature_extraction as numerical_FE
from classification.numerical.random_forest.classifier import RandomForestClassifier


def classify(mp3_path):
    features = numerical_FE.extract_features(mp3_path)
    classifier = RandomForestClassifier()
    prediction = classifier.classify([features])
    # Add the other 2 algorithms,
    # make an ensemble facade also that does the voting logic
    return prediction

def train(classifier_name, features_file_path, save_dir_path):
    if classifier_name == 'Random Forest':
        classifier = RandomForestClassifier()
    elif classifier_name == 'Convolutional Neural Network':
        pass  # Add logic for 2nd classifier
    elif classifier_name == 'Other':
        pass  # Add logic for 3rd item

    classifier.train(features_file_path, save_dir_path)
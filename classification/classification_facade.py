from feature_extraction.numerical import numerical_feature_extraction as numerical_fe
from classification.numerical.random_forest.classifier import RandomForestClassifier


def classify(mp3_path):
    features = numerical_fe.extract_features(mp3_path)
    classifier = RandomForestClassifier()
    prediction = classifier.classify([features])
    # Add the other 2 algorithms,
    # make an ensemble facade also that does the voting logic
    return prediction

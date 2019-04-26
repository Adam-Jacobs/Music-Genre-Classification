from classification.common.classifier_template import ClassifierTemplate
from common.label_manipulation import LabelManipulator
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


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
            return "Unclassifiable"

        genre_predictions_categorized = [x[1] for x in mlb.inverse_transform(result)]  # this needs checkinf for which value o fthe tuple is the actual value

        genre_predictions = []
        lm = LabelManipulator()
        for label in genre_predictions_categorized:
            genre_predictions.append(lm.uncategorise_genre(label))

        # convert the ids to names

        return genre_predictions
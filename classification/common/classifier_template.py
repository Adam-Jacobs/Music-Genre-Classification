class ClassifierTemplate:
    def __init(self):
        pass

    '''Returns 1-dimensional list of strings representing genre labels
    Parameters:
        features - 1-dimensional list of floats
    '''
    def classify(self, features):
        raise Exception('This method should be overwritten in inheriting classes')

    '''Trains a machine learning model, saving the files containing the trained model
    Parameters:
        features_file_path - the file path of a .csv file containing records in the following format:
                             [track_id, feature1, feature2, ..., featureN]
                             where N represents the number of features per track
        save_dir_path - path to the directory the trained model will be saved to
    '''
    def train(self, features_file_path, save_dir_path):
        raise Exception('This method should be overwritten in inheriting classes')

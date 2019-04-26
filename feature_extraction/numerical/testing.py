import numpy as np
import pandas as pd
import pickle

# Read in labels
pickle_in = open("..\\..\\dataset labels\\pickles\\training_labels.pickle", "rb")
dataset_labels = pickle.load(pickle_in)

pickle_in = open("..\\..\\dataset labels\\pickles\\testing_labels.pickle", "rb")
dataset_labels.extend(pickle.load(pickle_in))

pickle_in = open("..\\..\\dataset labels\\pickles\\validation_labels.pickle", "rb")
dataset_labels.extend(pickle.load(pickle_in))

# Read in features
features = pd.read_csv("data\\features.csv")

if len(dataset_labels) != len(features):
    print('Features has ' + str(len(dataset_labels) - len(features)) + ' missing records')
    print(len(dataset_labels))
    print(len(features))
    # x[x.columns[0]]
else:
    print('Correct features loaded')

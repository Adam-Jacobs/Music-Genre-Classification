import data_manipulation as dm
import numpy as np


features = [[1.0, 12.0],
            [2.0, 27.0],
            [3.0, 34.0]]
correct_normalised_values = [[0.0, 0.5, 1.0],
                             [0.0, 0.6818, 1.0]]

normalised_values = dm.normalise_features(features)
print(np.array(normalised_values))

import numpy as np
import os
import tqdm

features = []

for file_path in tqdm.tqdm(os.listdir("data")):
    features.extend(np.genfromtxt("data\\" + file_path,
                                  dtype=None, delimiter=','))

np.savetxt("data\\features.csv", features, delimiter=",", fmt="%s")

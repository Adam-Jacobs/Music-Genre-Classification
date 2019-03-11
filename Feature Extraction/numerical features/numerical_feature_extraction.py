import tqdm
import warnings

# Feature extraction imports
import librosa
import numpy as np

# Concurrency imports
from multiprocessing import Pool
from multiprocessing import Queue
from time import sleep
import gc

# File path retrieval imports
import os
import sys
sys.path.append("..\\common")
import utils


def extract_features(track_path):
    features = []
    y, sr = librosa.load(track_path)

    # Track Title
    features.append(os.path.splitext(os.path.basename(track_path))[0])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(zcr.sum() / zcr.size)

    # Mel Frequency Cepstral Coefficient
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    for i in range(0, 20):
        features.append(mfccs[i].sum() / mfccs[i].size)

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(cent[0].sum() / cent[0].size)

    gc.collect()

    return features


if __name__ == '__main__':
    track_paths = utils.get_files("Q:\\fma_full")

    resume_interval = 100

    for i in range(len(os.listdir("data")) * resume_interval, len(track_paths), resume_interval):
        tracks = []

        with Pool(os.cpu_count() - 2) as pool:
            tracks.extend(tqdm.tqdm(pool.imap(extract_features, track_paths[i:i+resume_interval]), total=resume_interval))

        gc.collect()

        np.savetxt("data\\features" + str(int(i / resume_interval)) + ".csv", tracks, delimiter=",", fmt="%s")

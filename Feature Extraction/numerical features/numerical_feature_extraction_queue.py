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
sys.path.append("..\\..\\common")
import utils


percentage = 0


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

    extract_features.queue.put(features)
    print('BIP')
    return features


def extract_features_init(queue):
    extract_features.queue = queue


if __name__ == '__main__':
    track_paths = utils.getFiles("Q:\\fma_full")

    tracks_queue = Queue()
    tracks = []

    pool = Pool(os.cpu_count(), extract_features_init, [tracks_queue])
    tracks.extend(pool.map(extract_features, track_paths[:20]))

    #while True:
        #if tracks_queue.empty() is False:
            #tracks.append(tracks_queue.get())

    pool.terminate()

    print('tracks length:' + str(len(tracks)))

    np.savetxt("data\\features.csv", tracks, delimiter=",", fmt="%s")

# Spectrogram generation imports
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Concurrency imports
from multiprocessing import Pool
import gc

# File path retrieval imports
import os
import sys
sys.path.append("..\\..\\common")
import utils


def generate_spectrogram(track_path):
    y, sr = librosa.load(track_path)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap='gray_r', y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(os.path.basename(track_path) + ' - mel spectrogram')
    plt.tight_layout()
    plt.savefig("images\\" + os.path.splitext(os.path.basename(track_path))[0] + ".png")
    plt.close()
    gc.collect()


if __name__ == '__main__':
    track_paths = utils.getFiles("Q:\\fma_full")
    pool = Pool(os.cpu_count())
    pool.map(generate_spectrogram, track_paths)
    pool.terminate()

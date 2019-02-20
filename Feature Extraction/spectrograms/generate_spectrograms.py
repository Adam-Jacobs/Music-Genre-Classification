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
sys.path.append("..\\common")
import utils


spectrograms_directory = "..\\..\\..\\..\\..\\..\\FYP_Data\\spectrogram_images"


def generate_spectrogram(track_path):
    y, sr = librosa.load(track_path)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap='gray_r', y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(os.path.basename(track_path) + ' - mel spectrogram')
    plt.tight_layout()
    plt.savefig(spectrograms_directory + "\\resumed\\" + os.path.splitext(os.path.basename(track_path))[0] + ".png")
    plt.close()
    gc.collect()


def resume(track_paths):
    completed_track_paths = utils.get_file_names(spectrograms_directory)

    print('Previously completed spectrograms: ' + str(len(completed_track_paths)))

    incomplete_track_paths = []
    for _, path in enumerate(track_paths):
        if os.path.splitext(os.path.basename(path))[0] not in completed_track_paths:
            incomplete_track_paths.append(path)

    return incomplete_track_paths


if __name__ == '__main__':
    print('Getting music file paths...')
    track_paths = utils.get_files("Q:\\fma_full")
    print('Music files identified: ' + str(len(track_paths)))

    print('Checking previously created spectrograms...')
    track_paths = resume(track_paths)
    print('Number of spectrograms remaining to create: ' + str(len(track_paths)))

    pool = Pool(os.cpu_count())
    pool.map(generate_spectrogram, track_paths)
    pool.terminate()

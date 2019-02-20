# Spectrogram generation imports
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# File path retrieval imports
import os
import utils


def process_generateSpectrogram(trackPath):
    y, sr = librosa.load(trackPath)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap='gray_r', y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(os.path.basename(trackPath) + ' - mel spectrogram')
    plt.tight_layout()
    plt.savefig("images\\" + os.path.splitext(os.path.basename(trackPath))[0] + ".png")
    plt.close()


if __name__ == '__main__':
    trackPaths = utils.getFiles("Q:\\fma_full")
    for i in range(0, len(trackPaths)):
        process_generateSpectrogram(trackPaths[i])

import cv2
import pickle
import tqdm
from multiprocessing import Pool
from PIL import Image
import os

image_startX = 81
image_endX = 803
image_startY = 36
image_endY = 341
image_sizeX = image_endX - image_startX
image_sizeY = image_endY - image_startY
image_size = image_sizeX * image_sizeY
image_scaled_sizeX = int(image_sizeX / 3)
image_scaled_sizeY = int(image_sizeY / 3)


def scale_cv2(path):
    # Read the image into memory
    data_point = cv2.imread(path)

    # Crop to include only the spectrogram (thus removing the axis and labelling)
    data_point = data_point[image_startY:image_endY, image_startX:image_endX]
    cv2.imwrite(os.path.basename(path).split('.')[0] + "_before_cv2.png", data_point)

    # Scale the image down for better memory management
    data_point = cv2.resize(data_point, (image_scaled_sizeX, image_scaled_sizeY))
    cv2.imwrite(os.path.basename(path).split('.')[0] + "_after_cv2.png", data_point)


def scale_other(path):
    # Read the image into memory
    data_point = Image.open(path)

    # Crop to include only the spectrogram (thus removing the axis and labelling)
    data_point = data_point[image_startY:image_endY, image_startX:image_endX]
    imwrite(os.path.basename(path).split('.')[0] + "_before_other.png", data_point)

    # Scale the image down for better memory management
    data_point = data_point.thumbnail((image_scaled_sizeX, image_scaled_sizeY), Image.ANTIALIAS)
    imwrite(os.path.basename(path).split('.')[0] + "_after_other.png", data_point)


def scale_spectrograms():
    spectrograms_dir = "..\\..\\..\\..\\..\\..\\..\\FYP_Data\\spectrogram_images"
    spectrogram_file_paths = os.listdir(spectrograms_dir)
    spectrogram_file_paths = [os.path.join(spectrograms_dir, x) for x in spectrogram_file_paths]
    spectrogram_file_paths = spectrogram_file_paths[:4]

    for i in tqdm.tqdm(range(len(spectrogram_file_paths))):
        scale_cv2(spectrogram_file_paths[i])
        scale_other(spectrogram_file_paths[i])


scale_spectrograms()

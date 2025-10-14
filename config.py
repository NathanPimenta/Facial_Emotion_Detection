# config.py
import os
import cv2

# Get the absolute path of the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained Keras model file (must be in the root folder)
MODEL_PATH = os.path.join(BASE_DIR, '100_model_file.h5')

# Target image size that the model expects
TARGET_SIZE = (48, 48)

# Path to the OpenCV Haar Cascade file (must ALSO be in the root folder)
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
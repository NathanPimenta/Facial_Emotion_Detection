import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import random

new_model = tf.keras.models.load_model('Final_model_95p07.h5')
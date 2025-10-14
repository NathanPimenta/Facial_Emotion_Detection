# backend_init.py

import cv2
from tensorflow import keras
import os
from config import MODEL_PATH, HAAR_CASCADE_PATH # <-- Import from config

# Global variables to hold the model and detector
emotion_model = None
FACE_CASCADE = None

def load_model_and_detector():
    """Loads the emotion model and face detector into global variables."""
    global emotion_model, FACE_CASCADE
    
    if os.path.exists(MODEL_PATH) and os.path.exists(HAAR_CASCADE_PATH):
        print("Loading model and face detector...")
        emotion_model = keras.models.load_model(MODEL_PATH)
        FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        print("Model and detector loaded successfully.")
    else:
        print(f"Error: Could not find model at '{MODEL_PATH}' or cascade at '{HAAR_CASCADE_PATH}'")
        # Handle the error appropriately, maybe exit or raise an exception
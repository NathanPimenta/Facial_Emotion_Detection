"""
video_analyze.py

Usage:
  # This file contains the local functions necessary for the Streamlit app
  # to analyze video frames directly on its local process.
  # NOTE: This bypasses the Flask GPU API and runs the model locally.
"""
import time
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

# --- CONFIGURATION (Matches the input requirements of your model) ---
from config import MODEL_PATH, TARGET_SIZE, HAAR_CASCADE_PATH
from emotion_mapper import RAW_CLASSES as CLASSES


def sample_frames_from_video(video_path: str, num_samples: int) -> List[np.ndarray]:
    """Samples 'num_samples' frames evenly from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        # Fallback reading all frames if count is unavailable
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        frame_count = len(frames)
        if frame_count == 0: raise RuntimeError("No frames found in video")
        
    indices = np.linspace(0, frame_count - 1, num=num_samples, dtype=int)
    sampled = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            sampled.append(frame)
            
    cap.release()
    return sampled

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE, channels: int = 1) -> np.ndarray:
    """Preprocess frame for model: convert color space, resize, normalize."""
    if channels == 1:
        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        arr = resized.astype('float32') / 255.0
        # shape (H,W) -> (H,W,1) for Keras
        arr = np.expand_dims(arr, axis=-1)
        return arr
    else:
        # RGB (not used for the 48x48 model, but kept for robustness)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, target_size)
        arr = resized.astype('float32') / 255.0
        return arr


def predict_on_frames(frames: List[np.ndarray], model: tf.keras.Model) -> Tuple[np.ndarray, List[str]]:
        """
        Runs the Keras model on a list of sampled frames.
        
        NOTE: This is a simplified function that assumes face detection and cropping 
        happened before calling this function in the older structure. 
        For accuracy, the original implementation must handle detection here or upstream.
        Since the goal is to provide the file used in the past, this assumes preprocessing 
        is handled correctly based on the model's expected 48x48 grayscale input.
        """
        
        # Determine model input shape to decide on target size and channels
        try:
            input_shape = model.input_shape
            _, h, w, c = input_shape
            target_size = (int(h), int(w))
            channels = int(c)
        except Exception:
            target_size = TARGET_SIZE
            channels = 1 

        # 1. Detect and preprocess every face in the sampled frames
        processed_faces = []
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 3) 
            
            # Simplified for local analysis: take the largest face found in the frame
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                sub_face_img = gray[y:y+h, x:x+w]
                
                # Preprocess for model input
                processed_frame = preprocess_frame(sub_face_img, target_size=target_size, channels=channels)
                processed_faces.append(processed_frame[0]) # Remove batch dim (1) for stacking
        
        if not processed_faces:
             # Return empty arrays if no faces were processed
             return np.zeros((0, len(CLASSES))), ["No Face Detected"] * len(frames)
             
        # 2. Predict on the entire batch
        batch = np.stack(processed_faces, axis=0)
        preds = model.predict(batch, verbose=0)
        preds = np.asarray(preds)
        
        # 3. Get labels
        pred_labels = [CLASSES[int(np.argmax(p))] for p in preds]
        return preds, pred_labels

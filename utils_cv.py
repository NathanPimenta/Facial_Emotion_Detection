# --- utils_cv.py ---
# Contains common computer vision functions needed by the Flask API.

import cv2
import numpy as np
import base64
import io
import tempfile
from typing import List, Tuple, Optional, Any
from config import TARGET_SIZE, HAAR_CASCADE_PATH

# --- PREPROCESSING UTILITIES ---

def preprocess_face_for_model(face_roi: np.ndarray) -> np.ndarray:
    """
    Resizes, normalizes, and reshapes a single grayscale face ROI 
    for Keras model prediction input (1, H, W, 1).
    """
    # 1. Resize (to 48x48)
    resized = cv2.resize(face_roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # 2. Normalize (0.0 to 1.0)
    normalize = resized / 255.0
    
    # 3. Reshape: (1, 48, 48, 1) -> Batch, Height, Width, Channel
    return np.reshape(normalize, (1, TARGET_SIZE[0], TARGET_SIZE[1], 1))

def sample_frames_from_video(video_path: str, num_samples: int) -> List[np.ndarray]:
    """Samples 'num_samples' frames evenly from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fallback for videos that don't report frame count
    if frame_count <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        if not frames: raise RuntimeError("No frames found in video")
        frame_count = len(frames)
        
    indices = np.linspace(0, frame_count - 1, num=num_samples, dtype=int)
    sampled = []
    
    # Efficiently seek and read frames
    for idx in indices:
        if frame_count > 0:
             cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            sampled.append(frame)
            
    # If the initial frame_count was 0, but we read frames via loop,
    # ensure indices are correctly calculated based on the loop count.
    if frame_count == 0 and len(sampled) > 0:
        return sampled[:num_samples]

    cap.release()
    return sampled

def annotate_frame(frame: np.ndarray, 
                   face_rect: Optional[Tuple[int, int, int, int]], 
                   emotion_label: str, 
                   status_label: str) -> str:
    """
    Annotates a single color frame with bounding boxes and both labels (Raw Emotion and Status), 
    then encodes the result to base64 PNG.
    """
    
    raw_text = emotion_label
    status_text = status_label
    
    # Color for boxes based on Status (Simple BGR colors for CV2)
    # Green for High Focus, Orange/Red for Negative Stress
    if "High Focus" in status_text:
        status_color = (0, 200, 0) # Green
    elif "Attention" in status_text:
        status_color = (0, 165, 255) # Orange
    elif "Negative" in status_text:
        status_color = (0, 0, 255) # Red
    else:
        status_color = (150, 150, 150) # Gray for 'No Face'

    if face_rect is None:
        # Default text if no face is detected
        cv2.putText(frame, "NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    else:
        x, y, w, h = face_rect
        
        # 1. Bounding Box (Around the face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
        
        # 2. Raw Emotion Label (Bottom box)
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 150, 0), -1) 
        cv2.putText(frame, raw_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 3. Interview Status Label (Top box)
        cv2.rectangle(frame, (x, y - 50), (x + w, y), status_color, -1) 
        cv2.putText(frame, status_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Encode annotated frame to base64
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

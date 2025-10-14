# --- api.py (Flask Backend - Single Image Only) ---
import os
import io
import numpy as np
import base64
import cv2 
import traceback
from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any
import tempfile
import os

from config import MODEL_PATH, TARGET_SIZE, HAAR_CASCADE_PATH
from emotion_mapper import RAW_CLASSES, INTERVIEW_STATUS_MAP

LABELS_DICT = {i: label for i, label in enumerate(RAW_CLASSES)}

# Initialize Flask application
app = Flask(__name__)
emotion_model = None
FACE_CASCADE = None

# --- Model and Detector Loading ---
def load_model_and_detector():
    """Loads the Keras model, initializes the detector, and checks GPU."""
    global emotion_model, FACE_CASCADE
    
    # 1. GPU Verification
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}. Using acceleration.")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU detected. Running on CPU.")

    # 2. Detector Loading
    try:
        FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if FACE_CASCADE.empty():
            raise IOError("Haar Cascade XML file could not be loaded or is empty.")
        print("✅ Haar Cascade Face Detector loaded successfully.")
    except Exception as e:
        print(f"🛑 Error loading Haar Cascade: {e}. Face detection will fail.")

    # 3. Model Loading
    try:
        emotion_model = keras.models.load_model(MODEL_PATH)
        emotion_model.predict(np.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1], 1)), verbose=0)
        print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"🛑 Error loading model: {e}")

# --- PREPROCESSING AND ANNOTATION UTILITIES ---
def preprocess_face_for_model(face_roi):
    """Resizes, normalizes, and reshapes a single grayscale face ROI."""
    resized = cv2.resize(face_roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    normalize = resized / 255.0
    return np.reshape(normalize, (1, TARGET_SIZE[0], TARGET_SIZE[1], 1))

def annotate_frame(frame, face_rect, emotion_label, status_label):
    """Annotates a single color frame with the bounding box and BOTH labels."""
    raw_text = emotion_label
    status_text = status_label
    
    # BGR colors for CV2
    if "High Focus" in status_text:
        status_color = (0, 200, 0) # Green
    elif "Attention" in status_text:
        status_color = (0, 165, 255) # Orange
    elif "Negative" in status_text:
        status_color = (0, 0, 255) # Red
    else:
        status_color = (150, 150, 150) # Gray for 'No Face'

    if face_rect is None:
        cv2.putText(frame, "NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    else:
        x, y, w, h = face_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
        
        cv2.rectangle(frame, (x,y+h), (x+w, y+h+30), (0, 150, 0), -1) 
        cv2.putText(frame, raw_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (x, y - 50), (x + w, y), status_color, -1) 
        cv2.putText(frame, status_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def _sample_frames_from_video_file(video_path: str, num_samples: int) -> List[np.ndarray]:
    """Samples num_samples frames evenly from a saved video file path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Fallback to reading everything if frame count is not available
    if frame_count <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError("No frames found in video")
        indices = np.linspace(0, len(frames) - 1, num=num_samples, dtype=int)
        return [frames[i] for i in indices]

    indices = np.linspace(0, frame_count - 1, num=num_samples, dtype=int)
    sampled = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            sampled.append(frame)

    cap.release()
    return sampled

# Load model and detector at import so endpoints are ready for clients
load_model_and_detector()

# --- API ENDPOINT ---
@app.route('/predict_emotion', methods=['POST'])
def predict_and_annotate_image():
    if emotion_model is None or FACE_CASCADE is None:
        return jsonify({"error": "Backend components (Model/Detector) not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part found in request. Use 'file' as the key."}), 400

    image_file = request.files['file']

    try:
        image_bytes = image_file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image file."}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 3) 
        last_predicted_emotion = "No Face Detected"
        interview_status = "Awaiting Face"

        if len(faces) > 0:
            x, y, w, h = faces[0]
            sub_face_img = gray[y:y+h, x:x+w]
            
            processed_face = preprocess_face_for_model(sub_face_img)
            predictions = emotion_model.predict(processed_face, verbose=0)
            label_index = np.argmax(predictions[0])
            
            emotion_label = LABELS_DICT[label_index]
            last_predicted_emotion = emotion_label
            interview_status = INTERVIEW_STATUS_MAP.get(emotion_label, "Unknown Status")

            base64_encoded_image = annotate_frame(frame, faces[0], emotion_label, interview_status)
        else:
             base64_encoded_image = annotate_frame(frame, None, last_predicted_emotion, interview_status)

        return jsonify({
            "status": "success",
            "prediction": last_predicted_emotion, 
            "mapped_status": interview_status,
            "annotated_image": base64_encoded_image 
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal prediction error. Details: {str(e)}"}), 500


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Accepts a video file and returns sampled, annotated frames plus aggregated statistics.

    Expected form-data fields:
    - video_file: the uploaded file
    - num_frames: optional integer (defaults to 12)
    """
    if emotion_model is None or FACE_CASCADE is None:
        return jsonify({"error": "Backend components (Model/Detector) not loaded."}), 500

    if 'video_file' not in request.files:
        return jsonify({"error": "No 'video_file' part found in request. Use 'video_file' as the key."}), 400

    try:
        num_frames = int(request.form.get('num_frames', 12))
    except Exception:
        num_frames = 12

    video_file = request.files['video_file']

    # Save uploaded video to a temporary file for OpenCV to read
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1] if video_file.filename else '.mp4')
    try:
        video_file.save(tmp.name)
        tmp.close()

        frames = _sample_frames_from_video_file(tmp.name, num_frames)

        annotated_frames_b64: List[str] = []
        mapped_statuses: List[str] = []
        predicted_labels: List[str] = []
        all_probabilities: List[List[float]] = []
        prediction_counts: Dict[str, int] = {}

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 3)

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                sub_face_img = gray[y:y+h, x:x+w]
                processed = preprocess_face_for_model(sub_face_img)
                preds = emotion_model.predict(processed, verbose=0)
                label_index = int(np.argmax(preds[0]))
                raw_label = LABELS_DICT.get(label_index, 'Unknown')
                mapped_status = INTERVIEW_STATUS_MAP.get(raw_label, 'Unknown Status')

                annotated_b64 = annotate_frame(frame.copy(), (x, y, w, h), raw_label, mapped_status)

                annotated_frames_b64.append(annotated_b64)
                mapped_statuses.append(mapped_status)
                predicted_labels.append(raw_label)
                all_probabilities.append([float(x) for x in preds[0].tolist()])

                prediction_counts[mapped_status] = prediction_counts.get(mapped_status, 0) + 1
            else:
                # No face detected for this frame
                annotated_b64 = annotate_frame(frame.copy(), None, 'No Face Detected', 'No Face Detected')
                annotated_frames_b64.append(annotated_b64)
                mapped_statuses.append('No Face Detected')
                predicted_labels.append('No Face Detected')
                all_probabilities.append([])
                prediction_counts['No Face Detected'] = prediction_counts.get('No Face Detected', 0) + 1

        # Ensure MAPPED status class labels are present for Streamlit plotting (order not important)
        class_labels = list(set(list(prediction_counts.keys())))

        response = {
            'status': 'success',
            'frame_count': len(frames),
            'annotated_frames': annotated_frames_b64,
            'mapped_statuses': mapped_statuses,
            'predicted_labels': predicted_labels,
            'all_probabilities': all_probabilities,
            'prediction_counts': prediction_counts,
            'class_labels': class_labels
        }

        return jsonify(response)
    except Exception as e:
        print(f"Video analysis error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal video analysis error. Details: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# --- Default Route for Health Check ---
@app.route('/', methods=['GET'])
def root():
    return {"Message": "Emopulse API for face detection is now live"}

# --- RUN THE APP ---
if __name__ == '__main__':
    load_model_and_detector()
    app.run(debug=True, host='0.0.0.0', port=5000)
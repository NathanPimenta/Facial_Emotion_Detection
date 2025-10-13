import os
import io
import numpy as np
import base64
import cv2 
from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf

# --- CONFIGURATION (Ensure these paths and names are correct) ---
# NOTE: Place your model file in the same directory as this app.py file
MODEL_PATH = '100_model_file.h5'
TARGET_SIZE = (48, 48)
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
LABELS_DICT = {i: label for i, label in enumerate(CLASS_LABELS)}

# Initialize Flask application
app = Flask(__name__)
emotion_model = None # Global variable to hold the loaded Keras model

# --- FACE DETECTOR SETUP ---
try:
    # Attempt to load the Haar Cascade classifier data built into OpenCV
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if FACE_CASCADE.empty():
        raise IOError("Haar Cascade XML file could not be loaded or is empty.")
    print("✅ Haar Cascade Face Detector loaded successfully.")
except Exception as e:
    print(f"🛑 Error loading Haar Cascade: {e}. Face detection will fail.")
    FACE_CASCADE = None

# --- Model Loading and Environment Check ---
def load_model():
    """Loads the Keras model and checks GPU availability."""
    global emotion_model
    
    # 1. GPU Verification
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}. Using acceleration.")
        try:
            # Prevent TensorFlow from allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU detected. Running on CPU.")

    # 2. Model Loading
    try:
        emotion_model = keras.models.load_model(MODEL_PATH)
        # Force a dummy prediction to ensure the model is fully initialized before first use
        # This prevents slowdowns on the first API call
        emotion_model.predict(np.zeros((1, 48, 48, 1)), verbose=0)
        print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"🛑 Error loading model: {e}")

# --- IMAGE PREPROCESSING FOR PREDICTION ---
def preprocess_face_for_model(face_roi):
    """
    Applies the exact sequence of preprocessing steps used for training:
    Resize, Grayscale, Normalize, and Reshape.
    """
    
    # 1. Resize and convert to grayscale (The face_roi input should already be grayscale)
    # We use numpy array slicing for efficiency here.
    resized = cv2.resize(face_roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # 2. Normalize the pixel values (0.0 to 1.0)
    normalize = resized / 255.0
    
    # 3. Reshape for the model: (1, 48, 48, 1) -> Batch, Height, Width, Channel
    reshaped = np.reshape(normalize, (1, TARGET_SIZE[0], TARGET_SIZE[1], 1))
    
    return reshaped

# --- API ENDPOINT ---
@app.route('/predict_emotion', methods=['POST'])
def predict_and_annotate():
    if emotion_model is None or FACE_CASCADE is None:
        return jsonify({"error": "Backend components (Model/Detector) not loaded."}), 500

    # Check for the standardized 'file' key from Streamlit
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part found in request. Use 'file' as the key."}), 400

    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # 1. Read and Decode Image
        # Use io.BytesIO to allow reading the image file multiple times if needed
        image_stream = io.BytesIO(image_file.read())
        np_array = np.frombuffer(image_stream.read(), np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # 2. Convert to Grayscale and Detect Faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 3) 
        
        last_predicted_emotion = "No Face Detected" # Default status

        if len(faces) == 0:
            # If no face is found, annotate the original image with a message
            cv2.putText(frame, last_predicted_emotion, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for x, y, w, h in faces:
                # Extract the face region (ROI) from the grayscale frame for prediction
                sub_face_img = gray[y:y+h, x:x+w]
                
                # 3. Preprocessing and Prediction
                processed_face = preprocess_face_for_model(sub_face_img)
                predictions = emotion_model.predict(processed_face, verbose=0)
                label_index = np.argmax(predictions[0])
                emotion_label = LABELS_DICT[label_index]
                last_predicted_emotion = emotion_label # Update the final result

                # 4. Annotation (Draw on the original COLOR frame)
                # Draw the bounding box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)
                # Draw the background for the text label
                cv2.rectangle(frame, (x,y-40), (x+w,y), (50,50,255), -1)
                # Put the text label
                cv2.putText(frame, emotion_label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # 5. Encode the Annotated Image back to Base64
        _, buffer = cv2.imencode('.png', frame)
        base64_encoded_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Return the annotated image data
        return jsonify({
            "status": "success",
            "prediction": last_predicted_emotion, 
            "annotated_image": base64_encoded_image 
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        # Log the full traceback for debugging locally
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

# --- Default Route for Health Check ---
@app.route('/', methods=['GET'])
def root():
    return {"Message": "Emopulse API for face detection is now live"}

# --- RUN THE APP ---
if __name__ == '__main__':
    load_model()
    # Host on 0.0.0.0 for accessibility, debug=True for auto-reload
    app.run(debug=True, host='0.0.0.0', port=5000)
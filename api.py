import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow import keras
import tensorflow as tf

# --- CONFIGURATION ---
# The path where you saved your trained model.
MODEL_PATH = '100_model_file.h5'
# The target size used during training
TARGET_SIZE = (48, 48)
# The emotion labels matching your training order
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize Flask application
app = Flask(__name__)

# --- GPU/CUDA VERIFICATION ---
# Check GPU availability once on startup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}. Model will use GPU acceleration.")
    # You can configure memory growth to avoid allocating all memory at once
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Model will run on CPU.")

# Load the Keras Model
try:
    # Use the Keras API embedded in TensorFlow
    emotion_model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"🛑 Error loading model: {e}")
    emotion_model = None

# --- IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(image_file):
    """Loads, resizes, converts image to grayscale, and normalizes it."""
    
    # 1. Open the image file from the request
    img = Image.open(io.BytesIO(image_file.read()))
    
    # 2. Convert to grayscale (as trained) and resize
    img = img.convert('L') 
    img = img.resize(TARGET_SIZE)
    
    # 3. Convert PIL image to NumPy array
    img_array = np.array(img, dtype='float32')
    
    # 4. Normalize the pixel values (as done by ImageDataGenerator)
    img_array /= 255.0
    
    # 5. Reshape for the model: (1, 48, 48, 1)
    # The model expects a batch dimension (1) and a channel dimension (1 for grayscale)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if emotion_model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    # Ensure an image file was sent
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part found in request."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Preprocess the uploaded image
        processed_image = preprocess_image(image_file)
        
        # Make prediction (GPU is used here if available)
        predictions = emotion_model.predict(processed_image)
        
        # Get the predicted class index (highest probability)
        predicted_index = np.argmax(predictions[0])
        
        # Map the index to the human-readable label
        predicted_emotion = CLASS_LABELS[predicted_index]
        
        # Format the full probabilities (optional)
        probabilities = {label: float(predictions[0][i]) for i, label in enumerate(CLASS_LABELS)}

        return jsonify({
            "status": "success",
            "prediction": predicted_emotion,
            "probabilities": probabilities,
            "index": int(predicted_index)
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    # To run in production, use gunicorn or uWSGI. 
    # For local development/testing:
    app.run(debug=True, host='0.0.0.0', port=5000)

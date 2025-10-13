import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FLASK_BASE_URL = os.getenv('FLASK_URL')

if FLASK_BASE_URL is None:
    FLASK_API_URL = "http://error-url/predict_emotion"
else:
    FLASK_API_URL = FLASK_BASE_URL.rstrip('/') + '/predict_emotion' 

# --- CONFIGURATION FOR DISPLAY ---
FIXED_HEIGHT = 450 

# --- API Interaction Function (Minimal) ---
def process_image_via_api(uploaded_file):
    # ... (function body remains the same) ...
    file_tuple = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    files = {'file': file_tuple} 
    
    try:
        response = requests.post(FLASK_API_URL, files=files)
        response.raise_for_status() 
        response_data = response.json()
        
        status = response_data.get('status')
        prediction = response_data.get('prediction', 'Processing Complete')
        
        if status == 'success' or status == 'warning':
            st.session_state['last_prediction_label'] = prediction
            st.success(f"Backend Prediction: {prediction}")
            return response_data.get('annotated_image') 
        else:
             st.error(f"Backend returned an error: {response_data.get('error', 'Check server logs.')}")
             return None

    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Flask API at {FLASK_API_URL}. Ensure the backend is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: API returned a bad status code. {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API processing: {e}")
        return None

# --- Image Resizing Utility (Unchanged) ---
def resize_image_for_display(image_input, target_height, is_base64=False):
    if is_base64:
        image_bytes = base64.b64decode(image_input)
    else:
        image_bytes = image_input

    img = Image.open(io.BytesIO(image_bytes))
    
    width, height = img.size
    new_width = int(width * (target_height / height))
    
    resized_img = img.resize((new_width, target_height))
    
    buf = io.BytesIO()
    resized_img.save(buf, format="PNG") 
    return buf.getvalue()

# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Emopulse: Emotion Analysis", layout="wide")
    
    # Custom CSS for styling and alignment
    st.markdown(f"""
        <style>
        /* CRITICAL FIX: Ensure images and captions align correctly */
        /* Target the columns that hold the images and force content to start at the top */
        [data-testid="stVerticalBlock"] > div[data-testid="column"] {{
            align-items: flex-start !important;
        }}
        
        /* CRITICAL FIX: Vertically center the emotion box */
        div[data-testid="column"] > div:has(.emotion-box-centered) {{
            display: flex;
            flex-direction: column;
            justify-content: center; /* Center vertically */
            align-items: center; /* Center horizontally */
            height: {FIXED_HEIGHT + 70}px; /* Adjusted height to match image + caption space */
            min-height: {FIXED_HEIGHT + 70}px;
        }}
        
        /* Emotion Box Styling */
        .emotion-box-centered {{
            text-align: center;
            font-size: 1.5em; 
            font-weight: bold;
            color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            margin: auto;
        }}
        
        /* Color overrides based on content */
        .emotion-Happy {{ background-color: #5cb85c !important; }}
        .emotion-Neutral {{ background-color: #337ab7 !important; }}
        .emotion-Angry, .emotion-Fear, .emotion-Sad, .emotion-Disgust {{ background-color: #d9534f !important; }}

        /* General styling from previous steps */
        .stButton>button {{ width: 100%; border-radius: 20px; background-color: #4CAF50; color: white; font-weight: bold; border: 2px solid #4CAF50; }}
        .stFileUploader {{ border: 2px dashed #007bff; padding: 20px; border-radius: 10px; text-align: center; }}
        .stSubheader {{ text-align: center !important; margin-bottom: 0 !important; padding-bottom: 0 !important; }}
        h1, h2 {{ color: #E0E0E0; text-align: center; }}
        .image-container img {{ max-height: {FIXED_HEIGHT}px; width: auto; display: block; margin-left: auto; margin-right: auto; border-radius: 8px; }}
        .css-1r6dn7c {{ text-align: center; font-style: italic; font-size: 0.9em; }}
        .stAlert {{ display: none; }}
        </style>
        """, unsafe_allow_html=True)

    st.title("Emopulse: AI-Powered Emotion Analysis")
    st.markdown("---")

    # --- Uploader Section (Centered) ---
    col_left_up, col_center_up, col_right_up = st.columns([1, 2, 1])
    
    with col_center_up:
        uploaded_file = st.file_uploader("Upload Image for Analysis:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        with col_center_up:
            if st.button("Analyze Emotion"):
                with st.spinner('Sending data to API and awaiting annotated result...'):
                    st.session_state['annotated_image_data'] = process_image_via_api(uploaded_file)
                    st.session_state['original_bytes'] = uploaded_file.getvalue()
    
        if 'annotated_image_data' in st.session_state and st.session_state['annotated_image_data']:
            
            # --- Image Comparison Section ---
            
            col_left_img, col_center_wrapper, col_right_img = st.columns([1, 2, 1])

            with col_center_wrapper:
                # 3 INNER COLUMNS: [Image 1] [Emotion] [Image 2]
                # Adjust ratio to allocate more space to the central emotion box
                col1, col_emotion_display, col2 = st.columns([1.5, 1, 1.5]) 
                
                # --- Column 1: Original Image ---
                with col1:
                    st.subheader("Image Before Analysis")
                    resized_original = resize_image_for_display(st.session_state['original_bytes'], FIXED_HEIGHT)
                    st.image(resized_original, caption="Original Upload") 
                
                # --- Column 3: Annotated Result ---
                with col2:
                    st.subheader("Image After Analysis")
                    try:
                        # Resize the annotated image
                        resized_annotated_bytes = resize_image_for_display(
                            st.session_state['annotated_image_data'], 
                            FIXED_HEIGHT, 
                            is_base64=True
                        )
                        st.image(resized_annotated_bytes, caption="Processed Image with Emotion Annotation")
                    except Exception as e:
                        st.error(f"Error decoding or displaying image from API response: {e}")
                
                # --- Column 2: Emotion Display (Vertical Center) ---
                with col_emotion_display:
                    emotion_label = st.session_state.get('last_prediction_label', 'Analyzing...')
                    
                    emotion_css_class = f"emotion-{emotion_label}"
                    
                    # The content is now centered vertically by CSS targeting the column container
                    st.markdown(f'<div class="emotion-box-centered {emotion_css_class}">Detected: {emotion_label}</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    if not os.getenv('FLASK_URL'):
        st.error("FATAL: FLASK_URL environment variable is not set.")
    else:
        main()
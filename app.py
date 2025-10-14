import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dotenv import load_dotenv

# --- Import Mapped Class Names from the new mapper file ---
# This ensures consistency between the frontend and backend labels for plotting
from emotion_mapper import RAW_CLASSES, MAPPED_STATUS_CLASSES

# Load environment variables
load_dotenv()
FLASK_BASE_URL = os.getenv('FLASK_URL', "http://127.0.0.1:5000") # Default to local URL

# --- API Endpoints ---
IMAGE_API_URL = FLASK_BASE_URL.rstrip('/') + '/predict_emotion'
VIDEO_API_URL = FLASK_BASE_URL.rstrip('/') + '/analyze_video'

# --- CONFIGURATION FOR DISPLAY ---
FIXED_HEIGHT = 450

# --- API INTERACTION FUNCTIONS ---

def process_image_via_api(uploaded_file):
    """Sends single image to Flask API for face detection and emotion prediction."""
    file_tuple = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    files = {'file': file_tuple}

    try:
        response = requests.post(IMAGE_API_URL, files=files)
        response.raise_for_status()
        response_data = response.json()

        status = response_data.get('status')
        # We grab the MAPPED status for display (e.g., "High Focus / Engaged")
        mapped_status = response_data.get('mapped_status', 'Processing Complete')

        if status == 'success' or status == 'warning':
            st.session_state['last_prediction_label'] = mapped_status
            st.success(f"Backend Status: {mapped_status} (Raw: {response_data.get('prediction')})")
            return response_data.get('annotated_image')
        else:
             st.error(f"Backend returned an error: {response_data.get('error', 'Check server logs.')}")
             return None

    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Flask API at {IMAGE_API_URL}. Ensure the backend is running.")
        return None
    # --- START OF DEBUGGING CHANGE 1 ---
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: API returned a bad status code. {e}")
        try:
            # Try to get the detailed error from the backend's JSON response
            error_json = e.response.json()
            if 'details' in error_json:
                st.subheader("Backend Error Details:")
                st.code(error_json['details'], language='python')
            else:
                st.error("The backend returned an error but did not provide details.")
        except:
            # This runs if the error response wasn't valid JSON
            st.error("Could not retrieve or parse detailed error from the backend.")
        return None
    # --- END OF DEBUGGING CHANGE 1 ---
    except Exception as e:
        st.error(f"An unexpected error occurred during API image processing: {e}")
        return None

def process_video_via_api(video_file, num_frames):
    """Sends video file to the Flask API for frame sampling and prediction."""
    files = {'video_file': (video_file.name, video_file.getvalue(), video_file.type)}
    data = {'num_frames': num_frames}

    try:
        st.info(f"Sending video for analysis to API: {VIDEO_API_URL}")
        response = requests.post(VIDEO_API_URL, files=files, data=data)
        response.raise_for_status()

        response_data = response.json()

        if response_data.get('status') == 'success' or response_data.get('status') == 'warning':
            st.success(f"Analysis Complete! Sampled {response_data.get('frame_count', 'N')} frames.")
            return response_data

        st.error(f"Video API Error: {response_data.get('error', response_data.get('message', 'Unknown error'))}")
        return None

    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Flask API at {VIDEO_API_URL}. Ensure the backend is running.")
        return None
    # --- START OF DEBUGGING CHANGE 2 ---
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: API returned a bad status code. {e}")
        try:
            # Try to get the detailed error from the backend's JSON response
            error_json = e.response.json()
            if 'details' in error_json:
                st.subheader("Backend Error Details:")
                st.code(error_json['details'], language='python')
            else:
                st.error("The backend returned an error but did not provide details.")
        except:
            # This runs if the error response isn't valid JSON
            st.error("Could not retrieve or parse detailed error from the backend.")
        return None
    # --- END OF DEBUGGING CHANGE 2 ---
    except Exception as e:
        st.error(f"An unexpected error occurred during API video processing: {e}")
        return None


# --- UTILITIES ---

def resize_image_for_display(image_input, target_height, is_base64=False):
    """Resizes image input (bytes or base64) for display in Streamlit."""
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

def plot_video_results(results):
    """Plots the aggregated results from the API using MAPPED statuses."""
    counts = results['prediction_counts']
    class_labels = results['class_labels'] # These are the MAPPED statuses

    # Calculate probabilities based on processed faces count
    total_valid_frames = sum([v for k, v in counts.items() if k != 'No Face Detected'])

    prob_dict = {}
    for cls in class_labels:
        prob_dict[cls] = counts.get(cls, 0) / total_valid_frames if total_valid_frames > 0 else 0

    # Left: compact pie chart
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Aggregated Status Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(3.2, 3.2))

        labels_nonzero = [k for k in class_labels if k in counts and k != 'No Face Detected']
        sizes_nonzero = [counts[k] for k in labels_nonzero]

        if len(sizes_nonzero) == 0:
            st.warning("No valid face detections for plotting.")
        else:
            counts_nonzero = [counts.get(k) for k in labels_nonzero]
            ax_pie.pie(counts_nonzero, labels=labels_nonzero, autopct='%1.0f%%', startangle=140, textprops={'fontsize':9})
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

    # Right: summary card
    with right_col:
        st.subheader("Overall Feedback")
        if prob_dict and total_valid_frames > 0:
            # Aggregate mapped statuses for the overall summary
            dominant_status = max(prob_dict.items(), key=lambda x: x[1])
            dom_label, dom_prob = dominant_status[0], dominant_status[1]
            st.markdown(f"<div style='background:#f0f2f6; padding:12px; border-radius:8px; text-align:center;'>\
                         <h3 style='margin:6px 0 4px 0'>Most Frequent Status:</h3>\
                         <div style='font-size:22px; font-weight:700'>{dom_label} ({dom_prob:.2%})</div>\
                         <div style='margin-top:8px; font-size:12px; color:#333'>Total valid frames: {total_valid_frames}</div>\
                         </div>", unsafe_allow_html=True)

        st.write("\n")
        st.markdown("**Status Counts**")
        # Display Mapped Statuses and their counts
        for cls in class_labels:
            st.write(f"- {cls}: {counts.get(cls,0)}")


def display_frame_by_frame_review(results):
    """
    Displays the navigation and specific frame details provided by the API.
    Uses MAPPED statuses and RAW probabilities.
    """
    if 'annotated_frames' not in results or not results['annotated_frames']:
        st.error("Backend did not return annotated frames.")
        return

    frames = results['annotated_frames']
    mapped_labels = results['mapped_statuses'] # MAPPED status list
    raw_labels = results['predicted_labels'] # Raw emotion list
    frame_count = len(frames)

    st.markdown("---")
    st.subheader("Frame-by-Frame Prediction Review")
    st.markdown("Use the **slider** to step through the sampled frames and see the instantaneous feedback.")

    # Slider for frame navigation
    frame_index = st.slider(
        'Select Frame',
        min_value=0,
        max_value=frame_count - 1,
        value=0,
        step=1,
        format='Frame %d'
    )

    col_img, col_info = st.columns([2, 1])

    with col_img:
        current_frame_b64 = frames[frame_index]
        current_mapped_status = mapped_labels[frame_index]

        # Display the annotated frame
        resized_annotated_bytes = resize_image_for_display(
            current_frame_b64,
            FIXED_HEIGHT,
            is_base64=True
        )
        st.image(resized_annotated_bytes, caption=f"Frame {frame_index + 1} of {frame_count}")

    with col_info:
        # Display detailed prediction for the selected frame
        st.markdown(f"### Feedback for Frame {frame_index + 1}")

        # Display Mapped Status
        emotion_css_class = f"emotion-{current_mapped_status.split(' ')[0]}" # Use the first word (e.g., 'High') for styling
        st.markdown(f'<div style="text-align:center;"><div class="emotion-box-centered {emotion_css_class}">Status: {current_mapped_status}</div></div>', unsafe_allow_html=True)

        # Display all RAW probabilities for the current frame
        if 'all_probabilities' in results and len(results['all_probabilities']) > frame_index:
            probs = results['all_probabilities'][frame_index]
            st.markdown("---")
            st.markdown("**Raw Emotion Probability Breakdown**")

            # Combine RAW_CLASSES with their probabilities
            prob_df = {
                'Emotion': RAW_CLASSES,
                'Probability': [f"{p:.3f}" for p in probs]
            }

            st.dataframe(
                prob_df,
                hide_index=True,
                column_config={"Probability": st.column_config.ProgressColumn(
                    "Probability",
                    format="%.3f",
                    min_value=0,
                    max_value=1.0
                )},
                height=300
            )
            # Display Raw Emotion (for developer debugging/clarity)
            st.markdown(f"**Raw Model Output:** `{raw_labels[frame_index]}`")


# --- STREAMLIT APP LAYOUT ---
def main():
    st.set_page_config(page_title="Emopulse: Interview Status Analysis", layout="wide")

    # --- Inject Custom CSS ---
    st.markdown(f"""
        <style>
        /* General Layout Cleanup */
        .stTabs [data-testid="stVerticalBlock"] > div[data-testid="column"] {{
            align-items: flex-start !important;
        }}
        
        .emotion-box-centered {{
            text-align: center;
            font-size: 1.5em; 
            font-weight: bold;
            color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            margin: auto;
            min-width: 150px;
        }}
        
        /* Tab Styling */
        .stTabs button[data-baseweb="tab"] {{
            font-size: 1.2em;
            font-weight: bold;
            padding: 10px 20px;
        }}
        
        /* Color overrides based on MAPPED status (only using the first word for simple CSS targeting) */
        .emotion-High {{ background-color: #5cb85c !important; }} /* Green for High Focus */
        .emotion-Attention {{ background-color: #f0ad4e !important; }} /* Orange for Attention/Confusion */
        .emotion-Negative {{ background-color: #d9534f !important; }} /* Red for Negative Stress */
        .emotion-No {{ background-color: #444444 !important; }} /* Dark Gray for No Face Detected */


        </style>
        """, unsafe_allow_html=True)

    st.title("InterviewPulse: AI-Powered Candidate Feedback")
    st.markdown(f"**Backend API URL:** `{FLASK_BASE_URL}`", unsafe_allow_html=True)
    st.markdown("---")

    # --- TAB NAVIGATION ---
    tab_video, tab_image = st.tabs(["🎥 Video Analysis (Interview Status)", "🖼️ Single Image Prediction (Raw Emotion)"])

    # --- VIDEO ANALYSIS TAB ---
    with tab_video:
        st.header("Video Analysis Configuration")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            uploaded_video = st.file_uploader("Upload Video (mp4/mov) for GPU Analysis", type=["mp4", "mov", "avi", "mkv"])

        with col_v2:
            num_frames = st.number_input("Number of frames to sample", min_value=2, max_value=60, value=12)
            if st.button("Analyze Video via Flask API", key="analyze_video_btn"):
                if uploaded_video is None:
                    st.warning("Please choose a video file.")
                else:
                    with st.spinner(f'Sending {uploaded_video.name} to Flask backend for GPU analysis...'):
                        results = process_video_via_api(uploaded_video, num_frames)
                        if results:
                            st.session_state['video_results'] = results
                            st.session_state['video_analysis_successful'] = True

        # Display video analysis results if available
        if st.session_state.get('video_analysis_successful', False) and st.session_state.get('video_results'):

            # 1. Aggregate Plots
            st.markdown("---")
            st.subheader("Aggregated Interview Status Results")
            plot_video_results(st.session_state['video_results'])

            # 2. Frame-by-Frame Review
            if st.session_state['video_results'].get('annotated_frames'):
                display_frame_by_frame_review(st.session_state['video_results'])
            else:
                 st.warning("Frame-by-frame review not available. Check backend error logs.")


    # --- IMAGE PREDICTION TAB ---
    with tab_image:
        st.header("Single Image Prediction (Raw Emotion)")

        col_left_up, col_center_up, col_right_up = st.columns([1, 2, 1])

        with col_center_up:
            uploaded_file = st.file_uploader("Upload Image for Analysis:", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_file is not None:

            with col_center_up:
                if st.button("Analyze Emotion (Image)", key="analyze_image_btn"):
                    with st.spinner('Sending data to API and awaiting annotated result...'):
                        st.session_state['annotated_image_data'] = process_image_via_api(uploaded_file)
                        st.session_state['original_bytes'] = uploaded_file.getvalue()

            if 'annotated_image_data' in st.session_state and st.session_state['annotated_image_data']:

                st.markdown("---")
                st.subheader("Image Analysis Comparison")

                col_left_img, col_center_wrapper, col_right_img = st.columns([1, 2, 1])

                with col_center_wrapper:
                    col1, col_emotion_display, col2 = st.columns([1.5, 1, 1.5])

                    # --- Column 1: Original Image ---
                    with col1:
                        st.markdown("**Original Image**")
                        resized_original = resize_image_for_display(st.session_state['original_bytes'], FIXED_HEIGHT)
                        st.image(resized_original, caption="Original Upload")

                    # --- Column 3: Annotated Result ---
                    with col2:
                        st.markdown("**Annotated Result**")
                        try:
                            resized_annotated_bytes = resize_image_for_display(
                                st.session_state['annotated_image_data'],
                                FIXED_HEIGHT,
                                is_base64=True
                            )
                            st.image(resized_annotated_bytes, caption="Processed Image with Annotation")
                        except Exception as e:
                            st.error(f"Error decoding or displaying image from API response: {e}")

                    # --- Column 2: Emotion Display (Vertical Center) ---
                    with col_emotion_display:
                        # Display Mapped Status (e.g., High Focus / Engaged)
                        emotion_label = st.session_state.get('last_prediction_label', 'Analyzing...')
                        emotion_css_class = f"emotion-{emotion_label.split(' ')[0]}"

                        st.markdown(f'<div class="emotion-box-centered {emotion_css_class}">Status: {emotion_label}</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
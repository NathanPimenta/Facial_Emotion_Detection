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
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import tempfile
import cv2
import av

# --- Import Mapped Class Names from the new mapper file ---
from emotion_mapper import RAW_CLASSES, MAPPED_STATUS_CLASSES

# Load environment variables
load_dotenv()
FLASK_BASE_URL = os.getenv('FLASK_URL', "http://127.0.0.1:5000")

# --- API Endpoints ---
IMAGE_API_URL = FLASK_BASE_URL.rstrip('/') + '/predict_emotion'
VIDEO_API_URL = FLASK_BASE_URL.rstrip('/') + '/analyze_video'

# --- CONFIGURATION ---
FIXED_HEIGHT = 450

# --- API INTERACTION FUNCTIONS ---
def process_image_via_api(uploaded_file):
    file_tuple = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    files = {'file': file_tuple}
    try:
        response = requests.post(IMAGE_API_URL, files=files)
        response.raise_for_status()
        response_data = response.json()
        status = response_data.get('status')
        mapped_status = response_data.get('mapped_status', 'Processing Complete')
        if status in ['success', 'warning']:
            st.session_state['last_prediction_label'] = mapped_status
            st.success(f"Backend Status: {mapped_status} (Raw: {response_data.get('prediction')})")
            return response_data.get('annotated_image')
        else:
            st.error(f"Backend returned an error: {response_data.get('error', 'Check server logs.')}")
            return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API image processing: {e}")
        return None

def process_video_via_api(video_file, num_frames):
    files = {'video_file': (video_file.name, video_file.getvalue(), video_file.type)}
    data = {'num_frames': num_frames}
    try:
        st.info(f"Sending video for analysis to API: {VIDEO_API_URL}")
        response = requests.post(VIDEO_API_URL, files=files, data=data)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get('status') in ['success', 'warning']:
            st.success(f"Analysis Complete! Sampled {response_data.get('frame_count', 'N')} frames.")
            return response_data
        st.error(f"Video API Error: {response_data.get('error', response_data.get('message', 'Unknown error'))}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API video processing: {e}")
        return None

# --- UTILITIES ---
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

def plot_video_results(results):
    counts = results['prediction_counts']
    class_labels = results['class_labels']
    total_valid_frames = sum([v for k, v in counts.items() if k != 'No Face Detected'])
    prob_dict = {cls: counts.get(cls, 0)/total_valid_frames if total_valid_frames > 0 else 0 for cls in class_labels}
    left_col, right_col = st.columns(2, vertical_alignment="center")
    left_col1, right_col1 = st.columns(2, vertical_alignment="center")

    with left_col:
        st.subheader("Aggregated Status Distribution")
        labels_nonzero = [k for k in class_labels if k in counts and k != 'No Face Detected']
        sizes_nonzero = [counts[k] for k in labels_nonzero]
        if len(sizes_nonzero) == 0:
            st.warning("No valid face detections for plotting.")
        else:
            fig = go.Figure(data=[go.Pie(
                values=sizes_nonzero, labels=labels_nonzero, 
                textinfo='label+percent', textfont=dict(size=15), pull=[0.1 for _ in labels_nonzero]
            )])
            fig.update_layout(width=450, height=600)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with right_col:
        
        st.header("**Status Counts**")
        fig = go.Figure(data=[go.Bar(
            x=list(counts.keys()), y=list(counts.values()), marker_color="#D35920"
        )])
        fig.update_layout(width=450, height=300)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<br>", unsafe_allow_html=True)
    with left_col1:
        st.subheader("Emotion Distribution on Per Frame Basis")
        frame_list = [0] + [i for i in range(1, len(results['predicted_labels']) + 1)]
        emotion_list = [0] + [results['emotion_ratings'].get(i, -1) for i in results['predicted_labels']]

        fig = go.Figure(

            data=[
                go.Scatter(x=frame_list, y=emotion_list, mode="lines+markers",
                text=[f"Frame: {f} <br> Emotion: {e}" for f, e in zip(frame_list, ['No Face Detected'] + results['predicted_labels'])]),
            ]
        )
        fig.update_layout(
            xaxis_title="Frame Number",
            yaxis_title="Emotion Rating",
            width=450,
            height=600,
            yaxis=dict(

                tickmode='array',
                tickvals = list(results['emotion_ratings'].values()),
                ticktext = list(results['emotion_ratings'].keys())
            ),
            xaxis = dict(
                tickmode='array',
                tickvals = list(frame_list),
                ticktext = list(frame_list)
            )
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with right_col1:
        pass


def display_frame_by_frame_review(results):
    if 'annotated_frames' not in results or not results['annotated_frames']:
        st.error("Backend did not return annotated frames.")
        return
    frames = results['annotated_frames']
    mapped_labels = results['mapped_statuses']
    raw_labels = results['predicted_labels']
    frame_count = len(frames)
    st.markdown("---")
    st.subheader("Frame-by-Frame Prediction Review")
    frame_index = st.slider(
        'Select Frame', min_value=0, max_value=frame_count-1, value=0, step=1, format='Frame %d'
    )
    col_img, col_info = st.columns([2,1])
    with col_img:
        current_frame_b64 = frames[frame_index]
        current_mapped_status = mapped_labels[frame_index]
        resized_annotated_bytes = resize_image_for_display(current_frame_b64, FIXED_HEIGHT, is_base64=True)
        st.image(resized_annotated_bytes, caption=f"Frame {frame_index + 1} of {frame_count}")
    with col_info:
        st.markdown(f"### Feedback for Frame {frame_index + 1}")
        emotion_css_class = f"emotion-{current_mapped_status.split(' ')[0]}"
        st.markdown(f'<div style="text-align:center;"><div class="emotion-box-centered {emotion_css_class}">Status: {current_mapped_status}</div></div>', unsafe_allow_html=True)
        if 'all_probabilities' in results and len(results['all_probabilities']) > frame_index:
            probs = results['all_probabilities'][frame_index]
            prob_df = {'Emotion': RAW_CLASSES, 'Probability': [f"{p:.3f}" for p in probs]}
            st.dataframe(
                prob_df,
                hide_index=True,
                column_config={"Probability": st.column_config.ProgressColumn(
                    "Probability", format="%.3f", min_value=0, max_value=1.0
                )},
                height=300
            )
            st.markdown(f"**Raw Model Output:** `{raw_labels[frame_index]}`")

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Emopulse: Interview Status Analysis", layout="wide")
    st.markdown(f"""
        <style>
        .stTabs [data-testid="stVerticalBlock"] > div[data-testid="column"] {{ align-items: flex-start !important; }}
        .emotion-box-centered {{ text-align: center; font-size: 1.5em; font-weight: bold; color: white; border-radius: 8px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.4); margin: auto; min-width: 150px; }}
        .stTabs button[data-baseweb="tab"] {{ font-size: 1.2em; font-weight: bold; padding: 10px 20px; }}
        .emotion-High {{ background-color: #5cb85c !important; }}
        .emotion-Attention {{ background-color: #f0ad4e !important; }}
        .emotion-Negative {{ background-color: #d9534f !important; }}
        .emotion-No {{ background-color: #444444 !important; }}
        </style>
    """, unsafe_allow_html=True)

    st.title("InterviewPulse: AI-Powered Candidate Feedback")
    st.markdown("---")

    # --- Initialize all session_state keys safely ---
    for key in [
        'video_results', 'video_analysis_successful',
        'uploaded_webcam_video_for_analysis',
        'webcam_video_results', 'webcam_video_analysis_successful'
    ]:
        if key not in st.session_state:
            st.session_state[key] = {}

    tab_video, tab_image, tab_live_video_record = st.tabs([
        "🎥 Video Analysis (Interview Status)",
        "🖼️ Single Image Prediction (Raw Emotion)",
        "📹 Live Video Recording (Interview Status)"
    ])

    # --- VIDEO TAB ---
    with tab_video:
        st.header("Video Analysis Configuration")
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            uploaded_video = st.file_uploader("Upload Video (mp4/mov) for GPU Analysis",
                                            type=["mp4", "mov", "avi", "mkv"])

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

        st.markdown("---")
        st.subheader("Aggregated Interview Status Results")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.get('video_analysis_successful'):
            plot_video_results(st.session_state['video_results'])
            if st.session_state['video_results'].get('annotated_frames'):
                display_frame_by_frame_review(st.session_state['video_results'])
            else:
                st.warning("Frame-by-frame review not available. Check backend error logs.")

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.header("Quick potential metrics for judging")

            avg_conf = st.session_state['video_results'].get('avg_confidence_regain_time', 0)
            times_lost = st.session_state['video_results'].get('times_confidence_lost', 0)
            st.text(f"Average time to regain confidence: {round(avg_conf, 2)} s")
            st.text(f"Number of times the candidate lost confidence / got distracted: {times_lost} times")
        else:
            st.info("Upload and analyze a video to see results.")

    # --- LIVE VIDEO RECORD TAB ---
    with tab_live_video_record:
        class FrameRecorder(VideoProcessorBase):
            def __init__(self):
                self.frames = []

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frames.append(img)
                return frame  # display live feed

        webcam_container = st.container()
        with webcam_container:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                webrtc_ctx = webrtc_streamer(
                    key="webcam-recorder",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=FrameRecorder,
                    media_stream_constraints={
                        "video": True, "audio": False,
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 30}
                    },
                    async_processing=True
                )

        with webcam_container:
            if st.button("Save Recorded Video"):
                if webrtc_ctx.video_processor and webrtc_ctx.video_processor.frames:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    height, width = webrtc_ctx.video_processor.frames[0].shape[:2]
                    out = cv2.VideoWriter(tmp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
                    for frame in webrtc_ctx.video_processor.frames:
                        out.write(frame)
                    out.release()
                    st.success(f"Video recorded and saved! Path: {tmp_file.name}")
                    st.session_state['uploaded_webcam_video_for_analysis'] = tmp_file.name
                else:
                    st.warning("No frames captured yet. Make sure the webcam is streaming for a few seconds.")

            num_frames1 = st.number_input("Number of frames to sample for analysis", min_value=10, value=12)

            if st.session_state.get('uploaded_webcam_video_for_analysis') and st.button("Analyze Recorded Video"):
                with open(st.session_state['uploaded_webcam_video_for_analysis'], "rb") as f:
                    files = {"video_file": (os.path.basename(f.name), f, "video/mp4")}
                    data = {"num_frames": num_frames1}
                    with st.spinner(f"Sending recorded video to Flask API ({VIDEO_API_URL})..."):
                        try:
                            response = requests.post(VIDEO_API_URL, files=files, data=data)
                            response.raise_for_status()
                            results = response.json()
                            if results.get('status') in ['success', 'warning']:
                                st.session_state['webcam_video_results'] = results
                                st.session_state['webcam_video_analysis_successful'] = True
                                st.success("Recorded video analysis complete!")
                        except Exception as e:
                            st.error(f"Error sending recorded video: {e}")

            if st.session_state.get('webcam_video_analysis_successful') and st.session_state.get('webcam_video_results'):
                st.markdown("---")
                st.subheader("Aggregated Interview Status Results")
                st.markdown("<br>", unsafe_allow_html=True)
                plot_video_results(st.session_state['webcam_video_results'])
                if st.session_state['webcam_video_results'].get('annotated_frames'):
                    display_frame_by_frame_review(st.session_state['webcam_video_results'])
                else:
                    st.warning("Frame-by-frame review not available. Check backend error logs.")

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.header("Quick potential metrics for judging")
                avg_conf2 = st.session_state['webcam_video_results'].get('avg_confidence_regain_time', 0)
                times_lost2 = st.session_state['webcam_video_results'].get('times_confidence_lost', 0)
                st.text(f"Average time to regain confidence: {round(avg_conf2, 2)} s")
                st.text(f"Number of times, the candidate lost confidence / gets distracted: {times_lost2} times")
            else:
                st.info("Record a short video, save it, and then analyze it to see results.")


    # --- IMAGE TAB ---
    with tab_image:
        st.header("Single Image Prediction (Raw Emotion)")
        col_left_up, col_center_up, col_right_up = st.columns([1, 2, 1])
        with col_center_up:
            uploaded_file = st.file_uploader("Upload Image for Analysis:", type=["jpg","jpeg","png"], key="image_uploader")
        if uploaded_file is not None:
            with col_center_up:
                if st.button("Analyze Emotion (Image)", key="analyze_image_btn"):
                    with st.spinner('Sending data to API and awaiting annotated result...'):
                        st.session_state['annotated_image_data'] = process_image_via_api(uploaded_file)
                        st.session_state['original_bytes'] = uploaded_file.getvalue()
            if 'annotated_image_data' in st.session_state and st.session_state['annotated_image_data']:
                st.markdown("---")
                st.subheader("Image Analysis Comparison")
                col_left_img, col_center_wrapper, col_right_img = st.columns([1,2,1])
                with col_center_wrapper:
                    col1, col_emotion_display, col2 = st.columns([1.5,1,1.5])
                    with col1:
                        st.markdown("**Original Image**")
                        resized_original = resize_image_for_display(st.session_state['original_bytes'], FIXED_HEIGHT)
                        st.image(resized_original, caption="Original Upload")
                    with col2:
                        st.markdown("**Annotated Result**")
                        try:
                            resized_annotated_bytes = resize_image_for_display(
                                st.session_state['annotated_image_data'], FIXED_HEIGHT, is_base64=True
                            )
                            st.image(resized_annotated_bytes, caption="Processed Image with Annotation")
                        except Exception as e:
                            st.error(f"Error decoding or displaying image from API response: {e}")
                    with col_emotion_display:
                        emotion_label = st.session_state.get('last_prediction_label', 'Analyzing...')
                        emotion_css_class = f"emotion-{emotion_label.split(' ')[0]}"
                        st.markdown(f'<div class="emotion-box-centered {emotion_css_class}">Status: {emotion_label}</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
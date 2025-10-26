Emopulse — Facial Emotion / Interview Status Detection

Small local project that runs a Keras/TensorFlow emotion classifier and exposes a Flask API used by a Streamlit frontend. The repo contains training code, utilities for preprocessing and CV, and two main runtime entrypoints:

- `api.py` — Flask backend that serves image and video analysis endpoints
- `app.py` — Streamlit frontend (calls the Flask API)

Files of interest
- `100_model_file.h5` — trained Keras model (expected in repo root)
- `config.py` — central configuration (model and cascade paths, target size)
- `emotion_mapper.py` — mapping of raw model classes to human-facing statuses
- `utils_cv.py`, `video_analysis.py` — CV helpers and local video analysis code
- `train.py` — training script used to create `100_model_file.h5`
- `api.py` — Flask API (endpoints: `/predict_emotion` and `/analyze_video`)
- `app.py` — Streamlit UI

Requirements
- Python 3.10+ recommended
- See `requirements.txt` for the pinned packages used while developing

Quick setup

1) Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the backend (Flask)

The Flask API can be started directly for development:

```bash
python3 api.py
```

This starts the API on http://0.0.0.0:5000 by default.

Or use Gunicorn for production-like execution (single process example):

```bash
gunicorn -w 1 -b 0.0.0.0:5000 api:app
```

Run the Streamlit frontend

By default the Streamlit app expects the Flask backend to be running at the URL in the `FLASK_URL` environment variable or `http://127.0.0.1:5000`.

```bash
# Optionally set FLASK_URL before launching Streamlit
export FLASK_URL=http://127.0.0.1:5000
streamlit run app.py
```

API endpoints

1) POST /predict_emotion
- Accepts form-data with key `file` containing an image (jpg/png).
- Response JSON (success):
  - `status`: "success"
  - `prediction`: raw emotion label (e.g., "Happy")
  - `mapped_status`: mapped interview feedback (e.g., "High Focus / Engaged")
  - `annotated_image`: base64-encoded PNG of the annotated image

Example curl:

```bash
curl -X POST "http://127.0.0.1:5000/predict_emotion" \
  -F "file=@/path/to/photo.jpg" \
  | jq .
```

2) POST /analyze_video
- Accepts form-data with key `video_file` and optional form field `num_frames` (integer, default 12). Returns sampled annotated frames and aggregated stats.
- Response JSON (success) example fields:
  - `status`: "success"
  - `frame_count`: number of frames sampled
  - `annotated_frames`: list of base64 PNG frames
  - `mapped_statuses`: list of mapped status strings per frame
  - `predicted_labels`: list of raw emotion labels per frame
  - `all_probabilities`: per-frame raw probability arrays
  - `prediction_counts`: aggregated counts of mapped statuses

Example curl (video):

```bash
curl -X POST "http://127.0.0.1:5000/analyze_video" \
  -F "video_file=@/path/to/video.mp4" \
  -F "num_frames=12" \
  | jq .
```

Developer notes & troubleshooting

- Missing packages: this project uses heavy dependencies (tensorflow, opencv-python, streamlit). If you see "No module named 'cv2'" or "No module named 'tensorflow'", install them into your active environment.

- Model and cascade must exist in the paths defined in `config.py`. If you want to place them elsewhere, update `config.py` or set absolute paths there.

- GPU: TensorFlow will print GPU device info at startup if available. The Flask backend will attempt to enable memory growth for the first GPU.

- Import-time initialization: by default `api.py` tries to load the model and the Haar cascade at import time (so endpoints are ready when the server starts). If you prefer lazy loading (on first request) we can change this to use a startup hook or Flask's `before_first_request` to avoid heavy imports during module import.

- Streamlit: `app.py` expects the backend to expose `/predict_emotion` and `/analyze_video`; the Streamlit UI will show helpful messages when the API is unreachable.

Testing locally

- Use `test.py` or `testdata.py` for quick experiments (they use `config.py` paths now). Those are simple scripts for webcam or single-image evaluation.


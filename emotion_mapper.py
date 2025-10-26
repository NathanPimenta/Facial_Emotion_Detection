# --- emotion_mapper.py ---

# The 7 raw emotion labels output by the model (MUST MATCH model's classes)
RAW_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- INTERVIEW FEEDBACK MAPPING ---
# Maps a single raw emotion to a categorized, actionable status.
# These will be the primary categories displayed in the Streamlit plots.
INTERVIEW_STATUS_MAP = {
    'Neutral': 'High Focus / Engaged',
    'Happy': 'High Focus / Engaged',
    
    'Surprise': 'Attention / Confused',
    'Fear': 'Attention / Confused',
    
    'Sad': 'Negative Stress / Disconnected',
    'Angry': 'Negative Stress / Disconnected',
    'Disgust': 'Negative Stress / Disconnected'
}

EMOTION_RATING={

    'Neutral': 6,
    'Happy': 7,
    
    'Surprise': 5,
    'Fear': 4,
    
    'Sad': 3,
    'Angry': 1,
    'Disgust': 2,
    'No Face Detected': 0
}

# The final list of MAPPED status names (for plotting/displaying in Streamlit)
MAPPED_STATUS_CLASSES = list(set(INTERVIEW_STATUS_MAP.values()))

# Utility function to quickly map a label
def map_emotion_to_status(emotion_label):
    return INTERVIEW_STATUS_MAP.get(emotion_label, "Unknown Status")

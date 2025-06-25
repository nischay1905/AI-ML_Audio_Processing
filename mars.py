import streamlit as st
import librosa
import numpy as np
import os
import pickle

# Get current script path
current_dir = os.path.dirname(__file__)

# Load model
model_path = os.path.join(current_dir, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load label encoder
encoder_path = os.path.join(current_dir, "label_encoder.pkl")
with open(encoder_path, "rb") as f:
    le = pickle.load(f)


# Function to extract MFCC from uploaded file
def extract_mfcc(file, n_mfcc=40):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

# Streamlit app UI
st.title("🎵 Speech Emotion Classifier")
st.markdown("Upload a `.wav` or `.mp3` file to predict the speaker's emotion.")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    try:
        features = extract_mfcc(audio_file)
        prediction = model.predict(features)
        emotion = le.inverse_transform(prediction)[0]
        st.success(f"🎤 Predicted Emotion: **{emotion.upper()}**")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
import streamlit as st
import librosa
import numpy as np
import pickle
import soundfile as sf

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Function to extract MFCC features
def extract_features(file):
    audio, sr = librosa.load(file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

# Streamlit app UI
st.title("🎧 Audio Classification App")
st.write("Upload a WAV file to classify it.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    try:
        features = extract_features(uploaded_file)
        prediction = model.predict(features)
        label = le.inverse_transform(prediction)[0]
        st.success(f"Predicted label: {label}")
    except Exception as e:
        st.error(f"Error: {e}")

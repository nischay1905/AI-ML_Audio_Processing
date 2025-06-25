import streamlit as st
import librosa
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as m:
    model = pickle.load(m)

# Load the label encoder
with open("label_encoder.pkl", "rb") as enc:
    label_encoder = pickle.load(enc)

# Function to extract MFCC features
def extract_features(audio):
    signal, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# Streamlit UI
st.title("🎵 Speech Emotion Classifier")
st.write("Upload a .wav or .mp3 file to predict the emotion.")

file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if file:
    st.audio(file, format="audio/wav")
    try:
        features = extract_features(file)
        pred = model.predict(features)
        emotion = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Emotion: *{emotion.upper()}*")
    except Exception as e:
        st.error(f"Could not process file: {e}")

# AI-ML-Audio-Processing
🎧 Emotion Recognition from Speech - Streamlit App
app-link = https://ai-mlaudioprocessing-iv9z5wyaammhyhhv74yqvw.streamlit.app/
This project is a complete machine learning pipeline to classify human emotions from speech using MFCC audio features. The trained model is deployed using Streamlit Cloud, allowing users to upload .wav or .mp3 files and get real-time emotion predictions.

 Project Description
The goal of this project is to classify human emotional states (like happy, sad, angry, etc.) using only speech audio. Users upload an audio file, and the app extracts MFCC features and uses a trained XGBoost model to predict the speaker's emotion.

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Modalities used: Audio (speech/song)

Classes: Calm, Happy, Sad, Angry, Fearful, Disgust

MFCC (Mel Frequency Cepstral Coefficients):
MFCCs are used to extract meaningful features from the audio signals. First, the audio time signal is transformed into a frequency signal using Fourier Transform. Then, Mel scaling is applied to emphasize lower frequencies, followed by taking the logarithm to simulate human hearing sensitivity (more responsive to lower frequencies). Finally, a Discrete Cosine Transform (DCT) is applied to produce a compact set of features called MFCC coefficients (typically 13–40). These features are then passed to the XGBoost classifier for emotion prediction.

📊 Accuracy Metrics
Overall Accuracy: ~78% on the test set

Macro F1-score: ~0.76

Confusion Matrix showed best performance for:

Neutral and Calm (high precision/recall)

Some confusion between Sad and Fearful (due to acoustic similarity)



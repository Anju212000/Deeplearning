import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
from bert import predict_sentiment
import sys
from io import StringIO


def record_audio(duration=5, sample_rate=16000):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    return np.squeeze(audio)

def save_audio(audio, filename, sample_rate=16000):
    write(filename, sample_rate, (audio * 32767).astype(np.int16))  # Save as 16-bit PCM WAV



def transcribe_audio(filename):
    st.info("Transcribing audio...")
    
    # Redirect sys.stderr to avoid tqdm issues
    original_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        model = whisper.load_model("small", device="cpu")
        result = model.transcribe(filename, language="en")
    finally:
        sys.stderr = original_stderr  # Restore original stderr

    return result["text"]

# Streamlit GUI
st.title("Sentiment analysis of audio using BERT")
duration = st.slider("Select recording duration (seconds)", 1, 10, 5)

if st.button("Start Recording"):
    audio = record_audio(duration)
    filename = "recorded_audio.wav"
    save_audio(audio, filename)
    st.audio(filename, format="audio/wav")
    transcription = transcribe_audio(filename)
    st.success(f"Transcription completed. Transcribed text is ` {transcription}` ")
    st.info("Analysing sentiment....")
    sentiment=predict_sentiment(transcription)
    st.write("Sentiment from the recorded audio is ", sentiment)
    st.success("sentiment analysis is completed ")


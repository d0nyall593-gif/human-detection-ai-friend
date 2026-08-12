import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
import streamlit as st
import cv2
import numpy as np
from fer import FER
from transformers import pipeline
import torch

st.title("👤 Human Detection + Emotion AI Friend")

# Auto-detect GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load models
emotion_detector = FER(mtcnn=True)
chat_model = pipeline("text-generation", model="distilgpt2", device=device)

# Chat loop
user_text = st.text_input("💬 Say something to your AI friend:")

if user_text:
    # Capture snapshot each time user talks
    camera_image = st.camera_input("📸 Take a picture while replying")

    if camera_image:
        # Convert image
        file_bytes = np.asarray(bytearray(camera_image.getbuffer()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Detect emotion
        emotions = emotion_detector.detect_emotions(img)
        dominant_emotion = "neutral"
        if emotions:
            dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)

        # Generate AI reply
        prompt = f"The person looks {dominant_emotion} and says: '{user_text}'. Respond kindly."
        reply = chat_model(prompt, max_length=80, do_sample=True)[0]['generated_text']

        st.write("🤖 Friend says:", reply)

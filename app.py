import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.title("Human Detection AI Friend (Webcam Mode)")

# Start webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Convert to RGB for display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show webcam feed
    FRAME_WINDOW.image(rgb_frame)

    try:
        # Analyze emotions in the current frame
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        st.write("Dominant emotion:", result[0]['dominant_emotion'])
    except Exception as e:
        st.write("No face detected or error:", e)

camera.release()


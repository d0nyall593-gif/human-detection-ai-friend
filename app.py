import streamlit as st
import numpy as np
from PIL import Image
from fer import FER

st.title("📸 Human Emotion Detector (Image Upload)")
st.write("Upload an image and let AI detect emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)

    if st.button("Analyze"):
        detector = FER(mtcnn=True)  # uses MTCNN face detector, no cv2 import
        result = detector.detect_emotions(img_array)

        if result:
            emotions = result[0]["emotions"]
            st.subheader("Results:")
            for emotion, score in emotions.items():
                st.write(f"{emotion}: {score:.2f}")
        else:
            st.warning("No face detected.")

from deepface import DeepFace
import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.title("Emotion Detection mit Statistik")

uploaded_file = st.file_uploader("Bild hochladen")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # enforce_detection=False verhindert Crash wenn kein Gesicht
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    emotions = result[0]["emotion"]
    df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Score"])
    st.image(img, channels="BGR")
    st.subheader("Emotion Statistik")
    st.bar_chart(df.set_index("Emotion"))
    st.write("Dominante Emotion:", result[0]["dominant_emotion"])

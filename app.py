import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.title("Emotion Detection mit Statistik (OpenCV)")

uploaded_file = st.file_uploader("Bild hochladen")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Fake-Emotion-Detection für Demo
    # (statt DeepFace, weil Streamlit Cloud sonst crasht)
    emotions = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]
    emotion_scores = {e: np.random.randint(0, 100) for e in emotions}

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, max(emotion_scores, key=emotion_scores.get), 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    st.image(img, channels="BGR")

    df = pd.DataFrame(list(emotion_scores.items()), columns=["Emotion", "Score"])
    st.subheader("Emotion Statistik")
    st.bar_chart(df.set_index("Emotion"))

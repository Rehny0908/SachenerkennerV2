import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

st.title("Echte KI: Emotion Detection (FER2013, Emo0.1)")

# 1️⃣ Modell laden von Hugging Face
@st.cache_resource
def load_emotion_model():
    model_path = hf_hub_download(
        repo_id="shivamprasad1001/Emo0.1",
        filename="facial_EmotionClassifer.h5"
    )
    return load_model(model_path)

model = load_emotion_model()
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 2️⃣ Bild-Upload
uploaded_file = st.file_uploader("Bild hochladen")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Gesichter erkennen
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=(0, -1))  # (1,48,48,1)

        preds = model.predict(roi_gray, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        results.append(preds)

        # Rechteck + Emotion Text
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    st.image(img, channels="BGR")

    # 4️⃣ Statistik: Durchschnitt aller Gesichter
    if results:
        avg_preds = np.mean(results, axis=0)
        df = pd.DataFrame(list(zip(emotion_labels, avg_preds*100)), columns=["Emotion", "Score"])
        st.subheader("Emotion Statistik (%)")
        st.bar_chart(df.set_index("Emotion"))
    else:
        st.write("Kein Gesicht erkannt!")

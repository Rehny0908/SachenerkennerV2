import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

st.title("Azure Face API Emotion Detection")

# 1️⃣ Azure Config
AZURE_KEY = st.secrets["AZURE_KEY"]       # aus Streamlit secrets
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
FACE_API_URL = AZURE_ENDPOINT + "/face/v1.0/detect"

# 2️⃣ Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img_array, caption="Hochgeladenes Bild", use_column_width=True)

    # 3️⃣ Bild für API vorbereiten
    _, img_encoded = cv2.imencode(".jpg", img_array)
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream"
    }
    params = {
        "returnFaceAttributes": "emotion",
        "returnFaceLandmarks": "false"
    }

    response = requests.post(FACE_API_URL, headers=headers, params=params, data=img_encoded.tobytes())
    
    if response.status_code != 200:
        st.error(f"API Fehler: {response.status_code}")
    else:
        faces = response.json()
        if not faces:
            st.write("Kein Gesicht erkannt!")
        else:
            # 4️⃣ Ergebnisse anzeigen
            all_emotions = []
            for face in faces:
                rect = face["faceRectangle"]
                emotions = face["faceAttributes"]["emotion"]
                all_emotions.append(emotions)
                
                dominant_emotion = max(emotions, key=emotions.get)
                st.write(f"Dominante Emotion: {dominant_emotion}")

                # Rechteck um Gesicht malen
                x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (255,0,0), 2)
            
            st.image(img_array, caption="Gesichter erkannt", use_column_width=True)

            # 5️⃣ Emotion Statistik
            df = pd.DataFrame(all_emotions)
            df = df.mean().sort_values(ascending=False).to_frame(name="Score")
            st.subheader("Emotion Statistik (%)")
            st.bar_chart(df)

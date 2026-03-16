import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("YOLO General Object Detection")

# YOLOv8 small model für generelle Objekte
model = YOLO("yolov8n.pt")  # erkennt >80 COCO-Klassen

uploaded_file = st.file_uploader("Bild hochladen")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    results = model.predict(source=img_array, verbose=False)

    annotated_frame = results[0].plot()  # Bounding Boxes + Labels
    st.image(annotated_frame, caption="Erkannte Objekte", use_column_width=True)

    # 1️⃣ Statistik: Anzahl erkannter Klassen
    counts = {}
    for box in results[0].boxes.cls.numpy():
        cls_name = model.names[int(box)]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    if counts:
        import pandas as pd
        df = pd.DataFrame(list(counts.items()), columns=["Objekt", "Anzahl"])
        st.subheader("Objekt Statistik")
        st.bar_chart(df.set_index("Objekt"))

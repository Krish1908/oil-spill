# D:\Coding\Infosys 6.0\new\src\ui.py

import streamlit as st
import requests
import base64
import io
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("AI-Driven Oil Spill Detection")
st.caption("Inference using Pre-Trained CNN & U-Net")
st.write(datetime.now().strftime("Timestamp: %d-%m-%Y %H:%M:%S IST"))
st.divider()

uploaded = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png", "tif"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                BACKEND_URL,
                files={"file": uploaded.getvalue()}
            )

        if response.status_code != 200:
            st.error("Backend error")
            st.stop()

        result = response.json()

        st.divider()
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Oil Spill Status",
                  "DETECTED" if result["oil_detected"] else "NOT DETECTED")

        c2.metric("Confidence", f"{result['confidence']} %")
        c3.metric("Oil Spill %", f"{result['oil_percentage']} %")
        c4.metric("Non-Spill %", f"{result['non_spill_percentage']} %")

        if result["oil_detected"]:
            mask = Image.open(
                io.BytesIO(base64.b64decode(result["mask_base64"]))
            )
            overlay = Image.open(
                io.BytesIO(base64.b64decode(result["overlay_base64"]))
            )

            st.divider()
            col1, col2, col3 = st.columns(3)

            col1.subheader("Original Image")
            col1.image(image)

            col2.subheader("Segmentation Mask")
            col2.image(mask)

            col3.subheader("Overlay Image")
            col3.image(overlay)

            st.subheader("Oil Spill Distribution")
            fig, ax = plt.subplots()
            ax.pie(
                [result["oil_percentage"],
                 result["non_spill_percentage"]],
                labels=["Oil Spill", "Non-Spill"],
                autopct="%1.1f%%",
                colors=["red", "green"]
            )
            ax.axis("equal")
            st.pyplot(fig)

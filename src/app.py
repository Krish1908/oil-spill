# D:\Coding\Infosys 6.0\new\src\app.py
# IMPORPER OVERLAY & UI ENHANCED — LOGIC UNCHANGED

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import io

# =================================================
# CONFIG
# =================================================
IMG_SIZE = 256

CNN_MODEL_PATH = r"D:\Coding\Infosys 6.0\new\models\cnn_classifier.keras"
UNET_MODEL_PATH = r"D:\Coding\Infosys 6.0\new\models\unet_model.keras"

CNN_THRESHOLD = 0.5
UNET_THRESHOLD = 0.4
MIN_AREA = 600

st.set_page_config(
    page_title="AI-Driven Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

# =================================================
# LOAD MODELS
# =================================================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    unet = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    return cnn, unet

cnn_model, unet_model = load_models()

# =================================================
# IMAGE PREPROCESSING (UNCHANGED)
# =================================================
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)

def land_water_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, water = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)
    return (water > 0).astype(np.uint8)

def postprocess_mask(pred_prob, image_rgb):
    raw_mask = (pred_prob > UNET_THRESHOLD).astype(np.uint8)
    oil_mask = 1 - raw_mask

    kernel = np.ones((5, 5), np.uint8)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(oil_mask)
    clean_mask = np.zeros_like(oil_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_AREA:
            clean_mask[labels == i] = 1

    img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    water_mask = land_water_mask(img_resized)

    return clean_mask & water_mask

def create_overlay(image_rgb, mask, alpha=0.4):
    image = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# =================================================
# HEADER
# =================================================
st.markdown("<h1 style='text-align:center'>🛢️ AI-Driven Oil Spill Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>CNN → Detection | U-Net → Segmentation</p>", unsafe_allow_html=True)

timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
st.caption(f"⏰ Analysis Timestamp: {timestamp}")

st.divider()

# =================================================
# SIDEBAR (RETRACTABLE)
# =================================================
with st.sidebar:
    st.header("⚙️ Detection Parameters")

    st.markdown("**Classification Stage (CNN)**")
    st.metric("Oil Detection Threshold", CNN_THRESHOLD)

    st.markdown("---")

    st.markdown("**Segmentation Stage (U-Net)**")
    st.metric("Segmentation Threshold", UNET_THRESHOLD)
    st.metric("Minimum Oil Region Area", MIN_AREA)

    st.markdown("---")

    st.markdown("**Legend**")
    st.markdown("""
    - 🟥 **Red Overlay** → Detected Oil Spill  
    - ⬜ **White** → Oil Region  
    - ⬛ **Black** → Non-Oil / Clean Water  
    """)


# =================================================
# MAIN UI
# =================================================
uploaded_file = st.file_uploader("Upload SAR / Satellite Image", type=["jpg","png","jpeg"])

if uploaded_file:
    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    img_proc = preprocess_image(img_rgb)
    img_batch = np.expand_dims(img_proc, axis=0)

    cnn_prob = cnn_model.predict(img_batch, verbose=0)[0][0]

    if cnn_prob < CNN_THRESHOLD:
        st.success("✅ No Oil Spill Detected")
        st.stop()

    st.warning("🚨 Oil Spill Detected")

    pred_prob = unet_model.predict(img_batch, verbose=0)[0].squeeze()
    final_mask = postprocess_mask(pred_prob, img_rgb)
    overlay = create_overlay(img_rgb, final_mask)

    oil_percent = (final_mask.sum() / final_mask.size) * 100
    water_percent = 100 - oil_percent

    # =================================================
    # PIE CHART
    # =================================================
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(
    [oil_percent, 100 - oil_percent],
    labels=["Oil Spill Area", "Clean Water Area"],
    autopct="%1.1f%%",
    colors=["#ff4d4d", "#4CAF50"],
    explode=(0.08, 0),
    startangle=90,
    textprops={"fontsize": 9}
    )
    ax.set_title("Oil Spill Area Distribution", fontsize=11)
    ax.axis("equal")



    st.pyplot(fig)
    plt.close(fig)


    # =================================================
    # VISUAL OUTPUT
    # =================================================
    col1, col2, col3 = st.columns(3)

    col1.image(img_rgb, caption="Original")
    col2.image(final_mask * 255, clamp=True, caption="Mask (White=Oil)")
    col3.image(overlay, caption="Overlay (Red=Oil)")

    # =================================================
    # CREATE DOWNLOAD IMAGE (MERGED)
    # =================================================
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    cv2.putText(
        overlay_bgr,
        f"Timestamp: {timestamp}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        overlay_bgr,
        "White=Oil | Black=No Oil | Red=Oil Region",
        (10, IMG_SIZE - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    buf_overlay = io.BytesIO()
    _, enc = cv2.imencode(".png", overlay_bgr)
    buf_overlay.write(enc)

    st.download_button(
        "⬇️ Download Overlay (with timestamp & legend)",
        buf_overlay.getvalue(),
        file_name="oil_spill_result.png",
        mime="image/png"
    )

st.divider()
st.caption("© AI-Driven Oil Spill Monitoring System")

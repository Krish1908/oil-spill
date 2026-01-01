# D:\Coding\Infosys 6.0\new\src\app.py
# IMPORPER OVERLAY & UI ENHANCED — LOGIC UNCHANGED

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime

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
# CUSTOM CSS (FROM UI CODE 1)
# =================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.6rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 1.5rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

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

def speckle_like_filter(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# =================================================
# LAND–WATER MASK (UNCHANGED)
# =================================================
def land_water_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, water = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7, 7), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)

    return (water > 0).astype(np.uint8)

# =================================================
# POST-PROCESSING (UNCHANGED)
# =================================================
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

# =================================================
# OVERLAY (UNCHANGED)
# =================================================
def create_overlay(image_rgb, mask, alpha=0.4):
    image = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# =================================================
# HEADER
# =================================================
st.markdown('<div class="main-header">🛢️ AI-Driven Oil Spill Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CNN → Detection | U-Net → Segmentation</div>', unsafe_allow_html=True)

current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
st.caption(f"⏰ Analysis Timestamp: {current_time}")

st.divider()

# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.metric("CNN Threshold", CNN_THRESHOLD)
    st.metric("U-Net Threshold", UNET_THRESHOLD)
    st.metric("Min Oil Area", MIN_AREA)

    st.divider()
    st.subheader("📘 About")
    st.info("""
    **Two-Stage Pipeline**
    • CNN detects presence  
    • U-Net segments spill  

    **Enhancements**
    • Land–Water masking  
    • Morphological cleanup  
    """)

# =================================================
# MAIN UI
# =================================================
st.header("📤 Upload Satellite Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(img_rgb, caption="📸 Uploaded Image", use_column_width=True)

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze = st.button("🔍 Analyze Image", type="primary")

    if analyze:
        with st.spinner("🔄 Analyzing image..."):
            img_proc = preprocess_image(img_rgb)
            img_batch = np.expand_dims(img_proc, axis=0)

            cnn_prob = cnn_model.predict(img_batch, verbose=0)[0][0]

            if cnn_prob < CNN_THRESHOLD:
                st.success("✅ NO OIL SPILL DETECTED")
                st.stop()

            st.warning("🚨 OIL SPILL DETECTED")

            pred_prob = unet_model.predict(img_batch, verbose=0)[0].squeeze()
            final_mask = postprocess_mask(pred_prob, img_rgb)
            overlay = create_overlay(img_rgb, final_mask)

            oil_percent = (final_mask.sum() / final_mask.size) * 100
            water_percent = 100 - oil_percent

            # Metrics
            st.subheader("📊 Detection Metrics")
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Status", "🔴 DETECTED")
            c2.metric("CNN Confidence", f"{cnn_prob*100:.2f}%")
            c3.metric("Oil Coverage", f"{oil_percent:.2f}%")
            c4.metric("Clean Water", f"{water_percent:.2f}%")

            st.divider()

            # Visuals
            st.subheader("🎨 Visual Analysis")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), caption="Original")

            with col2:
                st.image(final_mask * 255, clamp=True, caption="Segmentation Mask")

            with col3:
                st.image(overlay, caption="Overlay")

            st.divider()

            # Download
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", overlay_bgr)

            st.download_button(
                "⬇️ Download Overlay Image",
                buf.tobytes(),
                "oil_spill_overlay.png",
                mime="image/png"
            )

# # =================================================
# # FOOTER
# # =================================================
# st.divider()
# st.markdown("""
# <div style='text-align:center; color:#666; padding:2rem'>
# <strong>AI-Driven Oil Spill Detection System</strong><br>
# CNN + U-Net | Streamlit Deployment
# </div>
# """, unsafe_allow_html=True)

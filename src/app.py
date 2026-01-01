# D:\Coding\Infosys 6.0\new\src\app.py
# minor imperfections (LAND–WATER MASK ADDED ONLY)

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# =================================================
# CONFIG
# =================================================
IMG_SIZE = 256

CNN_MODEL_PATH = r"D:\Coding\Infosys 6.0\new\models\cnn_classifier.keras"
UNET_MODEL_PATH = r"D:\Coding\Infosys 6.0\new\models\unet_model.keras"

CNN_THRESHOLD = 0.5
UNET_THRESHOLD = 0.4
MIN_AREA = 600

st.set_page_config(page_title="AI-Driven Oil Spill Detection", layout="wide")

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
# IMAGE PREPROCESSING
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
# LAND–WATER MASK (ONLY ADDITION)
# =================================================
def land_water_mask(image_rgb):
    """
    Returns:
    water_mask = 1 for water, 0 for land
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Water is generally darker than land in SAR-like images
    _, water = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7, 7), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)

    return (water > 0).astype(np.uint8)

# =================================================
# POST-PROCESSING (UNCHANGED LOGIC)
# =================================================
def postprocess_mask(pred_prob, image_rgb):

    # 1️⃣ Threshold
    raw_mask = (pred_prob > UNET_THRESHOLD).astype(np.uint8)

    # 2️⃣ Invert (AS YOU HAD IT)
    oil_mask = 1 - raw_mask

    # 3️⃣ Morphology
    kernel = np.ones((5, 5), np.uint8)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel)

    # 4️⃣ Area filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(oil_mask)
    clean_mask = np.zeros_like(oil_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_AREA:
            clean_mask[labels == i] = 1

    # 🔑 5️⃣ APPLY LAND–WATER MASK (ONLY CHANGE)
    img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    water_mask = land_water_mask(img_resized)

    clean_mask = clean_mask & water_mask

    return clean_mask

# =================================================
# OVERLAY (UNCHANGED)
# =================================================
def create_overlay(image_rgb, mask, alpha=0.4):
    image = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# =================================================
# UI
# =================================================
st.title("🌊 AI-Driven Oil Spill Identification & Monitoring")
st.caption("CNN → Detection | U-Net → Segmentation")

uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(img_rgb, use_column_width=True)

    # Preprocess
    img_proc = preprocess_image(img_rgb)
    img_batch = np.expand_dims(img_proc, axis=0)

    # CNN
    cnn_prob = cnn_model.predict(img_batch, verbose=0)[0][0]
    st.write(f"**Oil Spill Probability:** `{cnn_prob:.3f}`")

    if cnn_prob < CNN_THRESHOLD:
        st.success("✅ No oil spill detected")
        st.stop()

    st.warning("⚠️ Oil spill detected — running segmentation")

    # U-Net
    pred_prob = unet_model.predict(img_batch, verbose=0)[0].squeeze()
    final_mask = postprocess_mask(pred_prob, img_rgb)
    overlay = create_overlay(img_rgb, final_mask)

    oil_percent = (final_mask.sum() / final_mask.size) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)))

    with col2:
        st.subheader("Final Oil Mask")
        st.image(final_mask * 255, clamp=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay)

    st.info(f"🛢 Estimated Oil Coverage: **{oil_percent:.2f}%**")

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    st.download_button("⬇️ Download Overlay", buf.tobytes(), "oil_spill_overlay.png")

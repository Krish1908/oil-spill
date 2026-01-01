# D:\Coding\Infosys 6.0\new\src\app.py
# UI ENHANCED — MODEL & OVERLAY LOGIC UNCHANGED

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

def create_overlay(image_rgb, mask, alpha=0.4, draw_outline=True, outline_color=(0,0,0)):
    image = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    overlay = image.copy()

    # Red oil overlay
    overlay[mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    if draw_outline:
        # Find contours of oil regions
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            blended,
            contours,
            -1,
            outline_color,
            thickness=2
        )

    return blended



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

st.markdown("""
<style>
/* Compact sidebar text */
.sidebar-compact h4 {
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.sidebar-compact p {
    font-size: 0.85rem;
    margin: 0.1rem 0;
    opacity: 0.9;
}

.sidebar-divider {
    margin: 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("🧭 System Configuration")

    st.markdown("<div class='sidebar-compact'>", unsafe_allow_html=True)

    st.markdown("#### Detection Thresholds")
    st.markdown(f"<p><b>CNN Threshold</b>: {CNN_THRESHOLD}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Segmentation Threshold</b>: {UNET_THRESHOLD}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Minimum Oil Area</b>: {MIN_AREA} pixels</p>", unsafe_allow_html=True)

    st.divider()

    st.markdown("**Legend**")
    st.markdown("""
    - 🟥 **Red Fill** → Oil Spill Region  
    - ⬜ **White Mask** → Oil (Segmentation Output)  
    - ⬛ **Black Mask** → Clean Water / Non-Oil  
    - ➖ **Outline** → Oil Boundary (toggleable)
    """)


    # =================================================
    # OUTLINE CONTROLS (UI ONLY)
    # =================================================
    st.markdown("---")
    st.subheader("🖍 Overlay Boundary")

    outline_enabled = st.toggle("Enable Oil Boundary Outline", value=True)

    outline_color_choice = st.radio(
        "Outline Color",
        ["Black (Day SAR)", "White (Night SAR)"],
        horizontal=True
    )

    OUTLINE_COLOR = (0, 0, 0) if outline_color_choice.startswith("Black") else (255, 255, 255)


# =================================================
# MAIN UI
# =================================================
uploaded_file = st.file_uploader(
    "Upload SAR / Satellite Image",
    type=["jpg", "png", "jpeg"]
)

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

    st.warning("🚨 Oil Spill Detected — Running Segmentation")

    pred_prob = unet_model.predict(img_batch, verbose=0)[0].squeeze()
    final_mask = postprocess_mask(pred_prob, img_rgb)
    overlay = create_overlay( img_rgb, final_mask, draw_outline=outline_enabled, outline_color=OUTLINE_COLOR)


    oil_percent = (final_mask.sum() / final_mask.size) * 100
    water_percent = 100 - oil_percent

    # =================================================
    # SMALL PIE CHART (OIL SPILL AREA)
    # =================================================
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [oil_percent, water_percent],
        labels=["Oil Spill", "Clean Water"],
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
    # VISUAL RESULTS
    # =================================================
    col1, col2, col3 = st.columns(3)
    col1.image(img_rgb, caption="Original Image")
    col2.image(final_mask * 255, clamp=True, caption="Mask (White = Oil)")
    col3.image(overlay, caption="Overlay (Red = Oil)")

    # =================================================
    # DOWNLOAD IMAGE (OVERLAY + PIE + TIMESTAMP)
    # =================================================
    fig_dl = plt.figure(figsize=(8, 4))

    ax1 = fig_dl.add_subplot(1, 2, 1)
    ax1.imshow(overlay)
    ax1.set_title("Oil Spill Overlay")
    ax1.axis("off")

    ax2 = fig_dl.add_subplot(1, 2, 2)
    ax2.pie(
        [oil_percent, water_percent],
        labels=["Oil Spill", "Clean Water"],
        autopct="%1.1f%%",
        colors=["#ff4d4d", "#4CAF50"],
        explode=(0.08, 0),
        startangle=90,
        textprops={"fontsize": 9}
    )
    ax2.set_title("Oil Spill Distribution", fontsize=10)
    ax2.text(
        0.5, -0.15,
        f"Timestamp: {timestamp}",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax2.transAxes
    )

    fig_dl.tight_layout()
    buf = io.BytesIO()
    fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_dl)
    buf.seek(0)

    safe_ts = timestamp.replace(":", "-").replace(" ", "_")

    st.download_button(
        "⬇️ Download Analysis Result",
        data=buf.getvalue(),
        file_name=f"oil_spill_analysis_{safe_ts}.png",
        mime="image/png"
    )

st.divider()

# nothing
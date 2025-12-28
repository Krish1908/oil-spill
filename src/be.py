# D:\Coding\Infosys 6.0\new\src\be.py - FIXED VERSION

import os
import io
import cv2
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_classifier.keras")
UNET_MODEL_PATH = os.path.join(MODELS_DIR, "unet_model.keras")

# ---------------- APP ----------------
app = FastAPI(title="Oil Spill Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODELS ----------------
print("🔄 Loading CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

print("🔄 Loading U-Net model...")
unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)

print("✅ Models loaded")

# ---------------- CONSTANTS ----------------
MODEL_SIZE = (256, 256)
CNN_THRESHOLD = 0.75  # Slightly lower for better detection
SEGMENT_THRESHOLD = 0.45  # Adjusted
MIN_OIL_AREA_RATIO = 0.01  # 1% (low to catch small spills)
MAX_OIL_AREA_RATIO = 0.85  # 85% max

# ---------------- HELPERS ----------------
def read_image(bytes_data):
    return np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))

def preprocess(img):
    img = cv2.resize(img, MODEL_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_mask(pred, original_shape):
    """
    FIXED: No inversion needed!
    The U-Net was trained with oil=1, water=0
    So we just threshold normally
    """
    # NO INVERSION - just threshold
    mask = (pred > SEGMENT_THRESHOLD).astype(np.uint8) * 255
    
    # Optional: Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Resize to original
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask

def create_overlay(img, mask):
    color = np.zeros_like(img)
    color[:, :, 2] = mask  # Red channel for oil spill
    return cv2.addWeighted(img, 0.85, color, 0.15, 0)

def encode_png(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

# ---------------- API ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image(image_bytes)

    # ---------- CNN ----------
    cnn_input = preprocess(image)
    cnn_score = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
    
    print(f"📊 CNN Score: {cnn_score:.4f}, Threshold: {CNN_THRESHOLD}")

    if cnn_score < CNN_THRESHOLD:
        print(f"❌ CNN rejected - Score too low")
        return {
            "oil_detected": False,
            "confidence": round((1 - cnn_score) * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None
        }

    print(f"✅ CNN passed - Proceeding to U-Net")

    # ---------- U-NET ----------
    unet_input = preprocess(image)
    pred_mask = unet_model.predict(unet_input, verbose=0)[0, :, :, 0]

    mask = postprocess_mask(pred_mask, image.shape)
    oil_ratio = np.sum(mask == 255) / mask.size
    
    print(f"📈 Oil ratio detected: {oil_ratio:.2%}")

    # Validation checks
    if oil_ratio < MIN_OIL_AREA_RATIO:
        print(f"❌ Oil area too small: {oil_ratio:.2%}")
        return {
            "oil_detected": False,
            "confidence": round(cnn_score * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None
        }
    
    if oil_ratio > MAX_OIL_AREA_RATIO:
        print(f"❌ Oil area too large (likely false positive): {oil_ratio:.2%}")
        return {
            "oil_detected": False,
            "confidence": round(cnn_score * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None
        }

    print(f"✅ Valid oil spill detected: {oil_ratio:.2%}")
    
    overlay = create_overlay(image, mask)
    oil_pct = round(oil_ratio * 100, 2)

    return {
        "oil_detected": True,
        "confidence": round(cnn_score * 100, 2),
        "oil_percentage": oil_pct,
        "non_spill_percentage": round(100 - oil_pct, 2),
        "mask_base64": encode_png(mask),
        "overlay_base64": encode_png(overlay)
    }

@app.get("/")
def health():
    return {"status": "Oil Spill Detection Backend Running"}

@app.get("/config")
def config():
    return {
        "cnn_threshold": CNN_THRESHOLD,
        "segment_threshold": SEGMENT_THRESHOLD,
        "min_oil_area_ratio": MIN_OIL_AREA_RATIO,
        "max_oil_area_ratio": MAX_OIL_AREA_RATIO
    }

# NEW CODE IS BEGIN GENERATED. START FROM SCRATCH
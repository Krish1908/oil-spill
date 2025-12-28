# D:\Coding\Infosys 6.0\new\src\be.py

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
CNN_THRESHOLD = 0.6
SEGMENT_THRESHOLD = 0.5
MIN_OIL_AREA_RATIO = 0.01  # 1%

# ---------------- HELPERS ----------------
def read_image(bytes_data):
    return np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))

def preprocess(img):
    img = cv2.resize(img, MODEL_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_mask(pred, original_shape):
    mask = (pred > SEGMENT_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask

def create_overlay(img, mask):
    color = np.zeros_like(img)
    color[:, :, 2] = mask
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

    if cnn_score < CNN_THRESHOLD:
        return {
            "oil_detected": False,
            "confidence": round((1 - cnn_score) * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None
        }

    # ---------- U-NET ----------
    unet_input = preprocess(image)
    pred_mask = unet_model.predict(unet_input, verbose=0)[0, :, :, 0]

    mask = postprocess_mask(pred_mask, image.shape)
    oil_ratio = np.sum(mask == 255) / mask.size

    if oil_ratio < MIN_OIL_AREA_RATIO:
        return {
            "oil_detected": False,
            "confidence": round(cnn_score * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None
        }

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
    return {"status": "Backend running"}

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

# ---------------- ADJUSTED CONSTANTS ----------------
MODEL_SIZE = (256, 256)

# INCREASED THRESHOLDS FOR MORE CONSERVATIVE DETECTION
CNN_THRESHOLD = 0.80          # Was 0.6 - Now requires 80% confidence
SEGMENT_THRESHOLD = 0.35      # Was 0.5 - Higher threshold for segmentation
MIN_OIL_AREA_RATIO = 0.02     # Was 0.01 - Now requires at least 5% of image
MAX_OIL_AREA_RATIO = 0.85     # NEW - Reject if > 85% is detected (likely false positive)

# ---------------- HELPERS ----------------
def read_image(bytes_data):
    """Read image from bytes"""
    return np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))

def preprocess(img):
    """Preprocess image for model input"""
    img = cv2.resize(img, MODEL_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_mask(pred, original_shape):
    """
    Enhanced post-processing with morphological operations
    to remove noise and clean up the mask
    """
    # INVERT THE PREDICTION
    pred = 1.0 - pred  # ADD THIS LINE

    # Apply threshold
    mask = (pred > SEGMENT_THRESHOLD).astype(np.uint8) * 255
    
    # Morphological operations to remove noise
    # Remove small white noise (opening)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small holes (closing)
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Additional erosion to reduce over-segmentation
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    # Resize to original shape
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    
    return mask

def create_overlay(img, mask):
    """Create overlay visualization with red highlighting"""
    color = np.zeros_like(img)
    color[:, :, 2] = mask  # Red channel
    return cv2.addWeighted(img, 0.85, color, 0.15, 0)

def encode_png(img):
    """Encode image as base64 PNG"""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

def validate_oil_detection(mask, original_shape):
    """
    Additional validation checks to reduce false positives
    Returns: (is_valid, reason)
    """
    oil_ratio = np.sum(mask == 255) / mask.size
    
    # Check 1: Too little oil detected
    if oil_ratio < MIN_OIL_AREA_RATIO:
        return False, f"Oil area too small: {oil_ratio*100:.2f}%"
    
    # Check 2: Too much oil detected (likely false positive)
    if oil_ratio > MAX_OIL_AREA_RATIO:
        return False, f"Oil area too large (likely false positive): {oil_ratio*100:.2f}%"
    
    # Check 3: Check if oil is in connected regions (not scattered everywhere)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Exclude background (label 0)
    if num_labels > 1:
        # Get sizes of all components (excluding background)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component = np.max(component_sizes)
        total_oil_pixels = np.sum(component_sizes)
        
        # If largest component is less than 30% of total oil pixels,
        # it means oil is too scattered (likely noise)
        if largest_component < 0.3 * total_oil_pixels:
            return False, "Oil detection too scattered (likely noise)"
    
    return True, "Valid oil spill detected"

# ---------------- API ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint with enhanced validation
    """
    image_bytes = await file.read()
    image = read_image(image_bytes)

    # ---------- STEP 1: CNN CLASSIFICATION ----------
    print(f"📊 CNN Threshold: {CNN_THRESHOLD}")
    cnn_input = preprocess(image)
    cnn_score = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
    print(f"🎯 CNN Score: {cnn_score:.4f}")

    if cnn_score < CNN_THRESHOLD:
        print(f"❌ CNN rejected: score {cnn_score:.4f} < threshold {CNN_THRESHOLD}")
        return {
            "oil_detected": False,
            "confidence": round((1 - cnn_score) * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None,
            "rejection_reason": "CNN confidence too low"
        }

    # ---------- STEP 2: U-NET SEGMENTATION ----------
    print(f"🔍 Segmentation Threshold: {SEGMENT_THRESHOLD}")
    unet_input = preprocess(image)
    pred_mask = unet_model.predict(unet_input, verbose=0)[0, :, :, 0]

    # Post-process mask with morphological operations
    mask = postprocess_mask(pred_mask, image.shape)
    
    # ---------- STEP 3: VALIDATION ----------
    is_valid, reason = validate_oil_detection(mask, image.shape)
    
    if not is_valid:
        print(f"❌ Validation failed: {reason}")
        return {
            "oil_detected": False,
            "confidence": round(cnn_score * 100, 2),
            "oil_percentage": 0.0,
            "non_spill_percentage": 100.0,
            "mask_base64": None,
            "overlay_base64": None,
            "rejection_reason": reason
        }

    # ---------- STEP 4: CALCULATE METRICS ----------
    oil_ratio = np.sum(mask == 255) / mask.size
    oil_pct = round(oil_ratio * 100, 2)
    
    print(f"✅ Oil detected: {oil_pct}% of image")
    print(f"📝 Reason: {reason}")

    # Create overlay
    overlay = create_overlay(image, mask)

    return {
        "oil_detected": True,
        "confidence": round(cnn_score * 100, 2),
        "oil_percentage": oil_pct,
        "non_spill_percentage": round(100 - oil_pct, 2),
        "mask_base64": encode_png(mask),
        "overlay_base64": encode_png(overlay),
        "detection_reason": reason
    }

@app.get("/")
def health():
    """Health check endpoint"""
    return {
        "status": "Backend running",
        "models_loaded": True,
        "thresholds": {
            "cnn_threshold": CNN_THRESHOLD,
            "segment_threshold": SEGMENT_THRESHOLD,
            "min_oil_ratio": MIN_OIL_AREA_RATIO,
            "max_oil_ratio": MAX_OIL_AREA_RATIO
        }
    }

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "cnn_threshold": CNN_THRESHOLD,
        "segment_threshold": SEGMENT_THRESHOLD,
        "min_oil_area_ratio": MIN_OIL_AREA_RATIO,
        "max_oil_area_ratio": MAX_OIL_AREA_RATIO,
        "model_input_size": MODEL_SIZE
    }
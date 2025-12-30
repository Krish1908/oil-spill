# /content/drive/MyDrive/Infosys 6.0/src/testing/test_unet.py

import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------
# ADD PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/Infosys 6.0/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# IMPORT METRICS
# ---------------------------------------------------
from models.unet import dice_coef

# ---------------------------------------------------
# PATH CONFIG (PROCESSED DATA)
# ---------------------------------------------------
MODEL_PATH = "/content/drive/MyDrive/Infosys 6.0/models/unet_model.keras"

TEST_IMG_DIR = "/content/drive/MyDrive/Infosys 6.0/src/data/processed/test/images"
TEST_MASK_DIR = "/content/drive/MyDrive/Infosys 6.0/src/data/processed/test/masks"

SAVE_DIR = "/content/drive/MyDrive/Infosys 6.0/models/unet_test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 256
THRESHOLD = 0.5

# 🔑 MUST MATCH TRAINING
WATER_CLASS = 174

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
print("📌 Loading trained U-Net model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"dice_coef": dice_coef}
)
print("✅ Model loaded successfully!")

# ---------------------------------------------------
# LOAD IMAGE + MASK (Numpy)
# ---------------------------------------------------
def load_image_mask(fname):
    img = np.load(os.path.join(TEST_IMG_DIR, fname)).astype("float32")
    mask = np.load(os.path.join(TEST_MASK_DIR, fname))

    # Convert multiclass → binary oil-spill mask
    mask = (mask != WATER_CLASS).astype(np.uint8)

    return img, mask

# ---------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    acc = accuracy_score(y_true_f, y_pred_f)
    prec = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    f1 = f1_score(y_true_f, y_pred_f, zero_division=0)

    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)

    union = np.sum((y_true_f + y_pred_f) > 0)
    iou = intersection / (union + 1e-6)

    return acc, prec, rec, f1, dice, iou

# ---------------------------------------------------
# MAIN TEST FUNCTION
# ---------------------------------------------------
def test_unet():
    img_files = sorted(os.listdir(TEST_IMG_DIR))

    all_acc, all_prec, all_rec = [], [], []
    all_f1, all_dice, all_iou = [], [], []

    print("📌 Running U-Net inference on test set...")

    for fname in img_files:
        img, gt_mask = load_image_mask(fname)

        pred = model.predict(img[None, ...], verbose=0)[0, ..., 0]
        pred_bin = (pred > THRESHOLD).astype(np.uint8)

        # ---------------- Visualization ----------------
        overlay = img.copy()
        overlay[pred_bin == 1] = [1, 0, 0]  # red overlay

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(pred_bin, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title("Overlay (Prediction)")
        plt.axis("off")

        save_path = os.path.join(SAVE_DIR, fname.replace(".npy", ".png"))
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # ---------------- Metrics ----------------
        acc, prec, rec, f1, dice, iou = compute_metrics(gt_mask, pred_bin)

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_dice.append(dice)
        all_iou.append(iou)

        print(f"✔ Saved: {save_path}")

    # ---------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------
    print("\n==============================")
    print("📊 U-NET TEST METRICS SUMMARY")
    print("==============================")
    print(f"Accuracy  : {np.mean(all_acc):.4f}")
    print(f"Precision : {np.mean(all_prec):.4f}")
    print(f"Recall    : {np.mean(all_rec):.4f}")
    print(f"F1 Score  : {np.mean(all_f1):.4f}")
    print(f"Dice Coef : {np.mean(all_dice):.4f}")
    print(f"IoU       : {np.mean(all_iou):.4f}")
    print("==============================\n")

    # ---------------------------------------------------
    # SHOW RANDOM SAMPLES
    # ---------------------------------------------------
    print("📌 Displaying 3 random predictions...\n")
    for fname in random.sample(img_files, 3):
        img = plt.imread(os.path.join(SAVE_DIR, fname.replace(".npy", ".png")))
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(fname)
        plt.show()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    test_unet()
    print("\n🎉 U-Net Testing Completed Successfully!")

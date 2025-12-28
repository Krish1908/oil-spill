# Infosys 6.0/new/src/testing/test_unet.py

import os
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------
# ADD PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = "Infosys 6.0/new/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# IMPORT U-NET METRICS
# ---------------------------------------------------
from models.unet import dice_coef

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
MODEL_PATH = "Infosys 6.0/new/models/unet_model.keras"
TEST_IMAGE_DIR = "Infosys 6.0/new/dataset_1/test/images"
TEST_MASK_DIR = "Infosys 6.0/new/dataset_1/test/masks"

SAVE_DIR = "Infosys 6.0/new/models/unet_test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 256
THRESHOLD = 0.5


# ---------------------------------------------------
# CUSTOM LOSS / METRIC DEFINITIONS
# ---------------------------------------------------
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
print("📌 Loading trained U-Net model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss
    }
)
print("✅ Model loaded successfully!")


# ---------------------------------------------------
# IMAGE + MASK LOADING
# ---------------------------------------------------
def load_image_mask(img_name):
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, img_name.replace(".jpg", ".png"))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 128).astype(np.uint8)

    return img, mask


# ---------------------------------------------------
# METRIC CALCULATIONS
# ---------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    acc = accuracy_score(y_true_f, y_pred_f)
    prec = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    f1 = f1_score(y_true_f, y_pred_f, zero_division=0)

    # Dice
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)

    # IoU
    union = np.sum((y_true_f + y_pred_f) > 0)
    iou = intersection / (union + 1e-6)

    return acc, prec, rec, f1, dice, iou


# ---------------------------------------------------
# MAIN TEST PROCESS
# ---------------------------------------------------
def test_unet():
    img_names = sorted(os.listdir(TEST_IMAGE_DIR))

    all_acc = []
    all_prec = []
    all_rec = []
    all_f1 = []
    all_dice = []
    all_iou = []

    print("📌 Running U-Net predictions and saving visual results...")

    for img_name in img_names:
        img, true_mask = load_image_mask(img_name)

        pred = model.predict(np.expand_dims(img, axis=0))[0]
        pred_bin = (pred > THRESHOLD).astype(np.uint8).squeeze()

        # Save visualization
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred.squeeze(), cmap="gray")
        plt.title("Predicted Mask (Raw)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_bin, cmap="gray")
        plt.title("Binary Mask")
        plt.axis("off")

        save_path = os.path.join(SAVE_DIR, img_name.replace(".jpg", ".png"))
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # Compute metrics
        acc, prec, rec, f1, dice, iou = compute_metrics(true_mask, pred_bin)

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_dice.append(dice)
        all_iou.append(iou)

        print(f"✔ Saved: {save_path}")

    print("\n==============================")
    print("📊 U-NET TEST METRICS SUMMARY")
    print("==============================")
    print(f"Accuracy      : {np.mean(all_acc):.4f}")
    print(f"Precision     : {np.mean(all_prec):.4f}")
    print(f"Recall        : {np.mean(all_rec):.4f}")
    print(f"F1 Score      : {np.mean(all_f1):.4f}")
    print(f"Dice Coef     : {np.mean(all_dice):.4f}")
    print(f"IoU           : {np.mean(all_iou):.4f}")
    print("==============================\n")

    # Show 3 random visualizations in output
    print("📌 Displaying 3 random predictions...\n")
    random_imgs = random.sample(img_names, 3)

    for name in random_imgs:
        display_path = os.path.join(SAVE_DIR, name.replace(".jpg", ".png"))

        img = plt.imread(display_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Visualization: {name}")
        plt.show()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    test_unet()
    print("\n🎉 U-Net Testing Completed!")

# /content/drive/MyDrive/Infosys 6.0/src/testing/test_cnn.py

import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/Infosys 6.0/src"
sys.path.append(PROJECT_ROOT)

TEST_IMG_DIR = "/content/drive/MyDrive/Infosys 6.0/src/data/processed/test/images"
TEST_MASK_DIR = "/content/drive/MyDrive/Infosys 6.0/src/data/processed/test/masks"
MODEL_PATH = "/content/drive/MyDrive/Infosys 6.0/models/cnn_classifier.keras"
SAVE_DIR = "/content/drive/MyDrive/Infosys 6.0/models/cnn_test_results"

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

def load_npy(path):
    return np.load(path)

# ---------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------
def load_test_data():
    img_files = sorted(os.listdir(TEST_IMG_DIR))

    X_test, y_test = [], []

    for f in img_files:
        img = load_npy(os.path.join(TEST_IMG_DIR, f))
        mask = load_npy(os.path.join(TEST_MASK_DIR, f))

        WATER_CLASS = 174  # same value used during training

        non_water_ratio = np.sum(mask != WATER_CLASS) / mask.size
        label = 1 if non_water_ratio > 0.05 else 0

        X_test.append(img)
        y_test.append(label)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    return img_files, X_test, y_test


# ---------------------------------------------------
# VISUALIZATION FUNCTION
# ---------------------------------------------------
def visualize_prediction(img, mask, true_label, pred_label, pred_prob, save_path):
    plt.figure(figsize=(10, 4))

    # -------- Input Image --------
    plt.subplot(1, 2, 1)
    plt.imshow((img * 255).astype(np.uint8))
    plt.title("Input Image")
    plt.axis("off")

    # -------- Ground Truth Mask --------
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # -------- Overall title --------
    plt.suptitle(
        f"Label: {true_label} | Pred: {pred_label} ({pred_prob:.2f})",
        fontsize=12
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    print("📌 Loading test data...")
    img_files, X_test, y_test = load_test_data()

    print("📌 Loading trained CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📌 Running predictions...")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_pred_prob)
    except:
        auc = 0.0

    print("\n============================")
    print("📊 CNN TEST RESULTS")
    print("============================")
    print(f"✔ Accuracy       : {acc:.4f}")
    print(f"✔ Precision      : {prec:.4f}")
    print(f"✔ Recall         : {rec:.4f}")
    print(f"✔ F1 Score       : {f1:.4f}")
    print(f"✔ AUC Score      : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("============================\n")

    # ---------------------------------------------------
    # SAVE VISUAL RESULTS FOR ALL IMAGES
    # ---------------------------------------------------
    print("📌 Saving visual results for all test samples...")

    for i, fname in enumerate(img_files):
        save_path = os.path.join(SAVE_DIR, f"pred_{fname.replace('.npy', '.png')}")
        visualize_prediction(
    X_test[i],
    load_npy(os.path.join(TEST_MASK_DIR, fname)),
    y_test[i],
    y_pred[i],
    y_pred_prob[i],
    save_path
)


    print("✅ All predictions saved!")

    # ---------------------------------------------------
    # SHOW VISUALIZATION FOR 3 RANDOM IMAGES
    # ---------------------------------------------------
    print("\n📌 Displaying 3 random visualizations...\n")

    random_indices = random.sample(range(len(X_test)), 3)

    for idx in random_indices:
      img = (X_test[idx] * 255).astype(np.uint8)
      mask = load_npy(os.path.join(TEST_MASK_DIR, img_files[idx]))

      plt.figure(figsize=(10, 4))

      plt.subplot(1, 2, 1)
      plt.imshow(img)
      plt.title("Input Image")
      plt.axis("off")

      plt.subplot(1, 2, 2)
      plt.imshow(mask, cmap="gray")
      plt.title("Ground Truth Mask")
      plt.axis("off")

      plt.suptitle(
          f"Label: {y_test[idx]} | Pred: {y_pred[idx]} ({y_pred_prob[idx]:.2f})"
      )

      plt.show()

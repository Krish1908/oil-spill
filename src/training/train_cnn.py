# Infosys 6.0/new/src/training/train_cnn.py

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ---------------------------------------------------
PROJECT_ROOT = "Infosys 6.0/new/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.dataloader import create_cnn_dataset
from models.cnn import build_cnn_model

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 40

MODEL_DIR = "Infosys 6.0/new/models"
MODEL_SAVE_PATH = f"{MODEL_DIR}/cnn_classifier.keras"
PLOT_SAVE_PATH = f"{MODEL_DIR}/cnn_training_plot.png"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_cnn():

    print("📌 Loading datasets...")
    train_ds = create_cnn_dataset("train", batch_size=BATCH_SIZE, augment=True)
    val_ds = create_cnn_dataset("val", batch_size=BATCH_SIZE, augment=False)

    print("📌 Building CNN model...")
    model = build_cnn_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    model.summary()

    # ---------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("🚀 Starting CNN Training on GPU...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # BEST EPOCH SUMMARY BLOCK
    # ---------------------------------------------------
    best_val_acc = max(history.history["val_accuracy"])
    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🏆 TRAINING SUMMARY")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.5f}")
    print(f"Best Validation AUC: {best_val_auc:.5f}")
    print(f"Model Saved At: {MODEL_SAVE_PATH}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # ---------------------------------------------------
    # SAVE MODEL + PLOT
    # ---------------------------------------------------
    print("💾 Saving final CNN model...")
    model.save(MODEL_SAVE_PATH)

    print("📊 Saving training performance plot...")
    plot_history(history)

    print("🎉 CNN Training Completed Successfully!")

# ---------------------------------------------------
# TRAINING PLOT
# ---------------------------------------------------
def plot_history(history):
    plt.figure(figsize=(10, 6))

    # Accuracy curves
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")

    # AUC curves
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")

    # Loss curves
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")

    plt.title("CNN Training Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(PLOT_SAVE_PATH)
    plt.close()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    train_cnn()

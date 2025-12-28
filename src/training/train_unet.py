# /content/drive/MyDrive/Infosys 6.0/src/training/train_unet.py

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# ADD PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/Infosys 6.0/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.dataloader import create_unet_dataset
from models.unet import build_unet, dice_coef, dice_loss

# ---------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------
MODEL_SAVE_PATH = "/content/drive/MyDrive/Infosys 6.0/models/unet_model.keras"
PLOT_SAVE_PATH = "/content/drive/MyDrive/Infosys 6.0/models/unet_training_plot.png"

os.makedirs("/content/drive/MyDrive/Infosys 6.0/models", exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 40

# ---------------------------------------------------
# COMBINED BCE + DICE LOSS
# ---------------------------------------------------
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    return bce + dloss

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
def train_unet():
    print("📌 Loading U-Net datasets...")
    train_ds = create_unet_dataset("train", batch_size=BATCH_SIZE, augment=True)
    val_ds = create_unet_dataset("val", batch_size=BATCH_SIZE, augment=False)

    model = build_unet()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True,
            monitor="val_accuracy", mode="max", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        )
    ]

    print("🚀 Training U-Net...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # SUMMARY BLOCK
    # ---------------------------------------------------
    best_epoch = history.history["val_accuracy"].index(max(history.history["val_accuracy"])) + 1
    best_acc = max(history.history["val_accuracy"])
    best_dice = max(history.history["dice_coef"])
    best_loss = min(history.history["val_loss"])
    
    print("\n============================")
    print("   📌 TRAINING SUMMARY")
    print("============================")
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"✔ Best Val Accuracy: {best_acc:.4f}")
    print(f"✔ Best Val Dice: {best_dice:.4f}")
    print(f"✔ Best Val Loss: {best_loss:.4f}")
    print("============================\n")

    print("💾 Saving final U-Net model...")
    model.save(MODEL_SAVE_PATH)

    print("📊 Saving training curve...")
    plot_history(history)

    print("🎉 U-Net Training Completed!")

# ---------------------------------------------------
# TRAINING PLOT
# ---------------------------------------------------
def plot_history(history):
    plt.figure(figsize=(10, 6))

    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")

    plt.plot(history.history["dice_coef"], label="Train Dice")
    plt.plot(history.history["val_loss"], label="Val Loss")

    plt.title("U-Net Training Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    train_unet()

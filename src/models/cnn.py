# Infosys 6.0/new/src/models/cnn.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(256, 256, 3)):
    """Optimized compact CNN for binary classification (Oil Spill = 1 / No Spill = 0)"""

    inputs = layers.Input(shape=input_shape)

    # Convolution Blocks
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs, name="OilSpill_CNN_Classifier")

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
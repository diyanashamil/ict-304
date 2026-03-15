"""
Subsystem B: CNN-based Flood Detection from Satellite Images
Train U-Net model on Sen1Floods11 dataset
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "satellite" / "v1.2"
TRAIN_CSV = DATA_DIR / "splits" / "flood_handlabeled" / "flood_train_data.csv"
VAL_CSV = DATA_DIR / "splits" / "flood_handlabeled" / "flood_val_data.csv"
S1_DIR = DATA_DIR / "data" / "flood_events" / "HandLabeled" / "S1Hand"
LABEL_DIR = DATA_DIR / "data" / "flood_events" / "HandLabeled" / "LabelHand"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparameters
IMG_SIZE = 256  # Resize images to 256x256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


def load_image_pair(s1_filename, label_filename):
    """Load and preprocess satellite image and flood mask."""
    # Load Sentinel-1 image (grayscale SAR)
    s1_path = S1_DIR / s1_filename
    img = cv2.imread(str(s1_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Image not found: {s1_path}")
    
    # Resize to IMG_SIZE x IMG_SIZE
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    
    # Ensure 3 channels (repeat grayscale)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # Load flood mask
    label_path = LABEL_DIR / label_filename
    mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {label_path}")
    
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    
    # Binary mask: 0 = no flood, 1 = flood
    # Assuming flood pixels are non-zero in the mask
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    
    return img, mask


def create_dataset(csv_path, batch_size=BATCH_SIZE):
    """Create TensorFlow dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    images = []
    masks = []
    
    print(f"Loading {len(df)} image pairs from {csv_path.name}...")
    
    for idx, row in df.iterrows():
        try:
            img, mask = load_image_pair(row['S1Hand'], row['LabelHand'])
            images.append(img)
            masks.append(mask)
            
            if (idx + 1) % 50 == 0:
                print(f"  Loaded {idx + 1}/{len(df)} images")
        except Exception as e:
            print(f"  Skipping {row['S1Hand']}: {e}")
            continue
    
    images = np.array(images)
    masks = np.array(masks)
    
    print(f"Dataset loaded: {len(images)} images, shape={images.shape}")
    
    # Create TF dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(images)


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build U-Net architecture for flood segmentation."""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (upsampling)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient for evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss for training."""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combined binary crossentropy + dice loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


if __name__ == "__main__":
    print("=" * 60)
    print("Subsystem B: CNN Flood Detection Training")
    print("=" * 60)
    
    # Load datasets
    train_dataset, train_size = create_dataset(TRAIN_CSV)
    val_dataset, val_size = create_dataset(VAL_CSV)
    
    print(f"\nTraining samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Build model
    print("\nBuilding U-Net model...")
    model = build_unet()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[
            dice_coefficient,
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "flood_cnn_best.keras",
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=5,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(MODEL_DIR / "flood_cnn_final.keras")
    print(f"\nModel saved to {MODEL_DIR / 'flood_cnn_final.keras'}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coefficient'], label='Train Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_history.png')
    print(f"Training plots saved to {MODEL_DIR / 'training_history.png'}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
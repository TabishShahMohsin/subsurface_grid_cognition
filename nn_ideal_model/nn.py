import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# === CONFIG ===
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = "grid_dataset"

# === LOAD LABELS ===
df = pd.read_csv(os.path.join(DATASET_DIR, 'labels.csv'))
df['filepath'] = df['filename'].apply(lambda f: os.path.join(DATASET_DIR, f))

# === LOAD IMAGES ===
def load_and_preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

X = np.array([load_and_preprocess(p) for p in df['filepath']])
y = df[['roll', 'pitch', 'yaw']].values.astype(np.float32)

# === SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL ===
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)  # roll, pitch, yaw
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === TRAIN ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

# === SAVE MODEL ===
model.save("pose_regressor_model.h5")

# === EVALUATE ===
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f"Validation MSE: {val_loss:.4f}, MAE: {val_mae:.2f}°")
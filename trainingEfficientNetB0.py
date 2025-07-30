import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Input, Lambda, Add, GlobalAveragePooling2D, Dropout, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_FOLDER = "/Users/laura/Desktop/FireDetectionImages"
CATEGORIES  = ['1','0']    # '1' = fire, '0' = no fire
SIZE        = (224,224)
BATCH       = 32
E_HEAD      = 5           # epochs for head training
E_FINE      = 10          # epochs for fine-tuning
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
AUTOTUNE    = tf.data.AUTOTUNE

# ─── COLLECT PATHS & LABELS ─────────────────────────────────────────────────
records = []
for label, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATA_FOLDER, cat)
    for fn in os.listdir(folder):
        if fn.lower().endswith(('.jpg','jpeg','.png')):
            records.append((os.path.join(cat, fn), label))
df = pd.DataFrame(records, columns=['file','label'])

# ─── PREPROCESSING FUNCTIONS ───────────────────────────────────────────────
def create_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary HSV mask of fire-colored regions."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (  0,120, 80), ( 10,255,255))
    m2 = cv2.inRange(hsv, (160,120, 80), (180,255,255))
    m3 = cv2.inRange(hsv, ( 10,120, 80), ( 40,255,255))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def segment_and_sharpen(bgr: np.ndarray) -> np.ndarray:
    """Segment fire regions then apply unsharp mask to sharpen."""
    mask = create_mask(bgr)
    seg  = cv2.bitwise_and(bgr, bgr, mask=mask)
    seg  = seg.astype('float32') / 255.0
    blur = cv2.GaussianBlur(seg, (0,0), sigmaX=3)
    return cv2.addWeighted(seg, 1.5, blur, -0.5, 0)

def load_raw_and_focus(path_rel: str) -> np.ndarray:
    """Load image, produce 6-channel array: [raw_rgb (3)] + [focus_rgb (3)]."""
    full = os.path.join(DATA_FOLDER, path_rel)
    img  = cv2.imread(full)
    img  = cv2.resize(img, SIZE)

    # 1) Raw RGB branch, preprocessed for EfficientNet
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    raw = preprocess_input(raw)

    # 2) Focus branch: segment & sharpen → back to 0–255 → preprocess
    focus = segment_and_sharpen(img)      # floats in [0,1]
    focus = (focus * 255).astype('float32')
    focus = preprocess_input(focus)

    # 3) Stack into shape (224,224,6)
    return np.concatenate([raw, focus], axis=-1)

# ─── LOAD INTO NUMPY ARRAYS ─────────────────────────────────────────────────
X = np.stack([
    load_raw_and_focus(f) for f in tqdm(df['file'], desc="Loading images")
]).astype(np.float32)
y = df['label'].values.astype(np.int32)

# ─── TRAIN/VAL SPLIT ────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── AUGMENTATION PIPELINE (tf.data) ────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

def make_dataset(X_arr, y_arr, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if training:
        ds = ds.shuffle(buffer_size=len(X_arr))
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val,   y_val,   training=False)

# ─── MODEL DEFINITION ───────────────────────────────────────────────────────
inp   = Input(shape=(*SIZE,6), name="six_channel_input")
raw   = Lambda(lambda x: x[..., :3], name="raw_rgb")(inp)
focus = Lambda(lambda x: x[..., 3:], name="fire_focus")(inp)

base   = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*SIZE,3))
feat_r = base(raw)
feat_f = base(focus)

merged = Add()([feat_r, feat_f])
gap    = GlobalAveragePooling2D()(merged)
drop   = Dropout(0.5)(gap)
out    = Dense(1, activation='sigmoid')(drop)

model = Model(inputs=inp, outputs=out)

# ─── PHASE 1: TRAIN HEAD ONLY ───────────────────────────────────────────────
for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(LR_HEAD),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, epochs=E_HEAD, validation_data=val_ds)

# ─── PHASE 2: FINE-TUNE TOP 20% ─────────────────────────────────────────────
cut = int(len(base.layers) * 0.8)
for layer in base.layers[cut:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(LR_FINE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, epochs=E_FINE, validation_data=val_ds)

# ─── FINAL EVALUATION & SAVE ───────────────────────────────────────────────
y_pred = (model.predict(X_val) > 0.5).astype(int).reshape(-1)
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification report:\n", classification_report(y_val, y_pred))

model.save("fire_detection_efficientnet6ch.keras")

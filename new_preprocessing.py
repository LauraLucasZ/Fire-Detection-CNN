import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

SIZE = (224, 224)

def create_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, (0,120,80),   (10,255,255))
    m2  = cv2.inRange(hsv, (160,120,80), (180,255,255))
    m3  = cv2.inRange(hsv, (10,120,80),  (40,255,255))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

def segment_and_sharpen(bgr: np.ndarray) -> np.ndarray:
    mask = create_mask(bgr)
    seg  = cv2.bitwise_and(bgr, bgr, mask=mask)
    seg  = seg.astype('float32') / 255.0
    blur = cv2.GaussianBlur(seg, (0,0), sigmaX=3)
    return cv2.addWeighted(seg, 1.5, blur, -0.5, 0)

def preprocess_6ch(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, SIZE)

    raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    raw = preprocess_input(raw)

    focus = segment_and_sharpen(img)
    focus = (focus * 255).astype('float32')
    focus = preprocess_input(focus)

    sixch = np.concatenate([raw, focus], axis=-1)
    return np.expand_dims(sixch, axis=0)

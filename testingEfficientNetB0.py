import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import keras  # note: this is the top-level keras that tf.keras re-exports
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─── ENABLE UNSAFE DESERIALIZATION ─────────────────────────────────────────
# This allows loading your Functional model with Lambda layers.
# WARNING: only do this if you trust your own saved .keras file!
keras.config.enable_unsafe_deserialization()

# ─── MODEL & CONSTANTS ──────────────────────────────────────────────────────
MODEL_PATH = "fire_detection_efficientnet6ch.keras"
SIZE       = (224, 224)

# Load the model (now succeeds despite Lambda layers)
model = tf.keras.models.load_model(MODEL_PATH)

# ─── PREPROCESSING (must match training) ────────────────────────────────────
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
    # 1) load & resize
    img = cv2.imread(image_path)
    img = cv2.resize(img, SIZE)

    # 2) raw RGB branch
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    raw = preprocess_input(raw)

    # 3) focus branch (segment+sharpen → back to 0–255 → preprocess)
    focus = segment_and_sharpen(img)           # floats [0,1]
    focus = (focus * 255).astype('float32')
    focus = preprocess_input(focus)

    # 4) stack into a single (1,224,224,6) batch
    sixch = np.concatenate([raw, focus], axis=-1)
    return np.expand_dims(sixch, axis=0)

# ─── GUI ───────────────────────────────────────────────────────────────────
class FireDetectionApp:
    def __init__(self, root):
        root.title("🔥 Fire Detection (EfficientNetB0) 🔥")
        root.configure(bg="#282c34")
        root.geometry("800x600")
        
        # 2) Prompt label (change color here)

        self.label = tk.Label(root,
            text="Choose an image to detect fire",
            font=("Arial",20), fg="#fee9be", bg="#282c34")
        self.label.pack(pady=10)

        style = ttk.Style(root)
        style.theme_use('clam')  # allows background customization
        style.configure(
            'Browse.TButton',
            font=('Helvetica', 16, 'bold'),
            foreground='#282c34',
            background='#FFFFFF',
            padding=(12, 8),
            borderwidth=0
        )
        style.map(
            'Browse.TButton',
            background=[('active', '#ddb5b6')],
            foreground=[('disabled', '#ddb5b6')]
        )

        self.btn = ttk.Button(
            root,
            text="📁 Browse Image",
            style='Browse.TButton',
            command=self.load_image
        )
        self.btn.pack(pady=5)

        self.image_panel = tk.Label(root, bg="#282c34")
        self.image_panel.pack(pady=10)

        self.result_label = tk.Label(root,
            text="", font=("Arial",18,"bold"),
            bg="#282c34")
        self.result_label.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.jpg *.jpeg *.png")])
        if not path:
            return

        # display thumbnail
        pil = Image.open(path).convert("RGB")
        pil.thumbnail((300,300))
        imgtk = ImageTk.PhotoImage(pil)
        self.image_panel.configure(image=imgtk)
        self.image_panel.image = imgtk

        # preprocess & predict
        try:
            x = preprocess_6ch(path)
        except Exception as e:
            messagebox.showerror("Preprocess Error", str(e))
            return

        p = model.predict(x)[0,0]
        if p > 0.5:
            text  = f"✅ No fire (Conf: {p:.2f})"
            color = "#4caf50"
        else:
            text  = f"🔥 Fire detected! (Conf: {1-p:.2f})"
            color = "#ff4c4c"

        self.result_label.configure(text=text, fg=color)

# ─── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    FireDetectionApp(root)
    root.mainloop()

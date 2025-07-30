import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# === Load VGG16 model ===
MODEL_PATH = "fire_detection_vgg16_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Preprocessing functions (same as training) ===
def create_mask(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0, 0, 250])
    upper_hsv = np.array([250, 255, 255])
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_image(image):
    mask = create_mask(image)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented / 255.0

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)

    # Apply segmentation and sharpening
    img_array = segment_image(img_array)
    img_array = sharpen_image(img_array)

    # VGG16 preprocessing
    img_array = preprocess_input(img_array.astype('float32'))

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Prediction Function ===
def predict_fire(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]
    prob = prediction[0]
    return (prob > 0.5), prob

# === GUI App ===
class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        root.title("🔥 Fire Detection with VGG16 🔥")
        root.geometry("600x600")
        root.configure(bg="#282c34")

        self.label = tk.Label(root, text="Select an image to detect fire", font=("Arial", 18), fg="#61dafb", bg="#282c34")
        self.label.pack(pady=20)

        self.btn = tk.Button(root, text="Choose Image", command=self.load_image, font=("Arial", 14), bg="#61dafb", fg="#282c34")
        self.btn.pack(pady=10)

        self.image_label = tk.Label(root, bg="#282c34")
        self.image_label.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 20, "bold"), bg="#282c34")
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        is_fire, prob = predict_fire(file_path)
        if is_fire:
            text = f"🔥 Fire detected! (Confidence: {prob:.2f})"
            color = "#ff4c4c"
        else:
            text = f"✅ No fire detected (Confidence: {1 - prob:.2f})"
            color = "#4caf50"

        self.result_label.configure(text=text, fg=color)

# === Main ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()

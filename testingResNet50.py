import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your model here - update the path to your preferred model file:
MODEL_PATH = "fire_detection_model.h5"  # You can switch to fire_detection.keras or the weights one if you have code to load weights separately

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)
# Recompile the model with the same settings as the training file
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Preprocessing function matching your training preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict fire or not
def predict_fire(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]
    # Assuming binary classification with one output neuron (sigmoid)
    if prediction.shape == (1,):  # Output shape (1,)
        prob = prediction[0]
    else:
        prob = prediction[1]  # If model outputs two classes probabilities
    if prob > 0.5:
        return True, prob
    else:
        return False, prob

# GUI application
class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        root.title("🔥 Fire Detection 🔥")
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

        # Load and display image
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        # Predict and display result
        is_fire, prob = predict_fire(file_path)
        if is_fire:
            text = f"🔥 Fire detected! (Confidence: {prob:.2f})"
            color = "#ff4c4c"
        else:
            text = f"✅ No fire detected (Confidence: {1-prob:.2f})"
            color = "#4caf50"

        self.result_label.configure(text=text, fg=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()

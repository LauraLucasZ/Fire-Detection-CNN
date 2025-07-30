import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Set data paths
data_folder = "/Users/laura/Desktop/FireDetectionImages"
categories = ['1', '0']
INPUT_SIZE = (224, 224)

# Initialize data storage
train_data = []

# Load and label data
for label, category in enumerate(categories):
    class_folder = os.path.join(data_folder, category)
    for file in os.listdir(class_folder):
        train_data.append(['{}/{}'.format(category, file), label])

# Create dataframe
df = pd.DataFrame(train_data, columns=['file', 'label'])

# Image preprocessing functions

def create_mask(image):
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define mask for bright areas
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


def preprocess_image(file_path):
    # Load and resize
    img = load_img(os.path.join(data_folder, file_path), target_size=INPUT_SIZE)
    img = img_to_array(img)
    # Apply segmentation and sharpening
    img = segment_image(img)
    img = sharpen_image(img)
    # Apply ResNet50 preprocessing
    img = preprocess_input(img)
    return img

# Data preparation
X = np.zeros((len(df), *INPUT_SIZE, 3), dtype='float32')
for i, file_path in tqdm(enumerate(df['file']), total=len(df)):
    X[i] = preprocess_image(file_path)

y = df['label'].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*INPUT_SIZE, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Save the entire model
model.save("fire_detection_vgg16_model.h5")

# Save only the model weights
model.save_weights("fire_detection_vgg16_weights.weights.h5")


# Save the model in the SavedModel format
model.save("fire_detection_vgg16.keras")


# Model training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluation
y_pred = (model.predict(X_val) > 0.5).astype('int32')
print(classification_report(y_val, y_pred))

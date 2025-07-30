import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image preprocessing functions

def create_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red range #1
    lower1 = np.array([  0, 120,  80])
    upper1 = np.array([ 10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Red range #2
    lower2 = np.array([160, 120,  80])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Orange/Yellow range
    lower3 = np.array([ 10, 120,  80])
    upper3 = np.array([ 40, 255, 255])
    mask3 = cv2.inRange(hsv, lower3, upper3)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_image(image):
    m = create_mask(image)
    seg = cv2.bitwise_and(image, image, mask=m)
    return seg / 255.0


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


def preprocess_image(path, target_size=(224,224)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = segment_image(img)
    img = sharpen_image(img)
    return img
import cv2
import matplotlib.pyplot as plt
from image_processing import create_mask, segment_image, sharpen_image, preprocess_image
import os

# Define the image path
image_path = '/Users/laura/Desktop/FireDetectionImages/0/19.jpg'  # Change this to any sample image path

# Load the image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply each function
mask = create_mask(image)
segmented = segment_image(image)
sharpened = sharpen_image(image)
preprocessed = preprocess_image(image_path)

# Plot the results
fig, axs = plt.subplots(1, 5, figsize=(20, 5))
axs[0].imshow(image_rgb)
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(mask, cmap='gray')
axs[1].set_title('Mask')
axs[1].axis('off')

axs[2].imshow(segmented)
axs[2].set_title('Segmented')
axs[2].axis('off')

axs[3].imshow(sharpened)
axs[3].set_title('Sharpened')
axs[3].axis('off')

axs[4].imshow(preprocessed)
axs[4].set_title('Preprocessed')
axs[4].axis('off')

plt.tight_layout()
plt.show()

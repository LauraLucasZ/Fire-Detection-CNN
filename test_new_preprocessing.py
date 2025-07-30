import cv2
import matplotlib.pyplot as plt
from new_preprocessing import create_mask, segment_and_sharpen, preprocess_6ch
import numpy as np

# Define the image path (change to a path on your system)
image_path = "/Users/laura/Desktop/FireDetectionImages/0/20.jpg"

# Load the image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply each function
mask = create_mask(image)
seg_sharp = segment_and_sharpen(image)
preprocessed_6ch = preprocess_6ch(image_path)[0]  # shape: (224, 224, 6)

# Split 6ch image into RGB and Focus channels
rgb_3ch = preprocessed_6ch[:, :, :3]
focus_3ch = preprocessed_6ch[:, :, 3:]

# De-process input for visualization
def deprocess_input(x):
    x = x.copy()
    x = x - np.min(x)
    x = x / np.max(x)
    return x

# Plot the results
fig, axs = plt.subplots(1, 5, figsize=(20, 5))
axs[0].imshow(image_rgb)
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(mask, cmap='gray')
axs[1].set_title('Mask')
axs[1].axis('off')

axs[2].imshow(seg_sharp)
axs[2].set_title('Segment+Sharpen')
axs[2].axis('off')

axs[3].imshow(deprocess_input(rgb_3ch))
axs[3].set_title('RGB Preprocessed')
axs[3].axis('off')

axs[4].imshow(deprocess_input(focus_3ch))
axs[4].set_title('Focus Preprocessed')
axs[4].axis('off')

plt.tight_layout()
plt.show()

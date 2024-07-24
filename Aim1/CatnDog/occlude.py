import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load original and perturbed images
# folder_path = "Both folder Path"  # Update with the actual folder path
num_images = 80  # Total number of images (original + perturbed)

original_confidence_changes = []
perturbed_confidence_changes = []

for i in range(1, num_images + 1):
    # Assume naming convention: original_1.jpg, original_2.jpg, ..., perturbed_1.jpg, perturbed_2.jpg, ...
    image_type = "original" if i <= 40 else "perturbed"
    image_path = f"{folder_path}/{image_type}_{i}.jpg"

    # Load image
    img = cv2.imread(image_path)

    # Compute confidence changes (you'll need your trained model here)
    # Example: Use a pre-trained CNN model to predict class probabilities
    # and compare before and after occlusion.

    # For demonstration purposes, let's assume some random confidence changes
    original_confidence_change = np.random.uniform(0.1, 0.5)
    perturbed_confidence_change = np.random.uniform(-0.2, 0.2)

    # Append confidence changes to lists
    if image_type == "original":
        original_confidence_changes.append(original_confidence_change)
    else:
        perturbed_confidence_changes.append(perturbed_confidence_change)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(1, num_images + 1), original_confidence_changes, label="Original", color="blue")
plt.scatter(range(1, num_images + 1), perturbed_confidence_changes, label="Perturbed", color="red")
plt.xlabel("Image Index")
plt.ylabel("Confidence Change")
plt.title("Occlusion Sensitivity: Original vs. Perturbed Images")
plt.legend()
plt.grid(True)
plt.show()

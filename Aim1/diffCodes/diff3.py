import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_images_from_folder(folder):
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            yield img

def plot_histogram(pixel_intensities):
    plt.figure(figsize=(10, 6))
    sns.histplot(pixel_intensities, bins=64, kde=True)  # Reduced bins
    plt.title('Aggregated Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Folder path containing images
folder = 'plcae your path here'

# Load images (using a generator)
images_generator = load_images_from_folder(folder)

# Aggregate pixel intensities from all images
pixel_intensities = np.concatenate([img.flatten() for img in images_generator])

# Plot histogram
plot_histogram(pixel_intensities)

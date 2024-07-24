import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))  # Sort to ensure matching order
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img) 
    return images

def aggregate_pixel_intensities(images):
    all_pixels = []
    for img in images:
        all_pixels.extend(img.flatten())
    return all_pixels

def plot_histogram(pixel_intensities):
    plt.figure(figsize=(10, 6))
    sns.histplot(pixel_intensities, bins=256, kde=True)
    plt.title('Aggregated Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def plot_heatmap(pixel_intensities, num_bins=256):
    # Calculate histogram
    hist, bins = np.histogram(pixel_intensities, bins=num_bins, range=(0, 256))
    hist = hist.reshape((1, -1))  # Reshape for heatmap

    plt.figure(figsize=(10, 1))
    sns.heatmap(hist, cmap='viridis', cbar=True, xticklabels=range(0, 256, 25), yticklabels=[])
    plt.title('Heatmap of Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.show()

# Folder path containing images
folder = 'plcae your path here'

# Load images
images = load_images_from_folder(folder)

# Aggregate pixel intensities from all images
pixel_intensities = aggregate_pixel_intensities(images)

# Plot histogram using Seaborn
plot_histogram(pixel_intensities)

# Plot heatmap using Seaborn
plot_heatmap(pixel_intensities)

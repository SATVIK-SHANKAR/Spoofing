import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#! modifying the below function to read only 20 images due to memory overflow
def load_images_from_folder(folder, batch_size=4):
    filenames = sorted(os.listdir(folder))
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i : i + batch_size]
        batch_images = [cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE) for filename in batch_filenames]
        yield batch_images
        count-=1

def aggregate_pixel_intensities(images):
    for img in images:
        yield from img.flatten()

def plot_histogram(pixel_intensities):
    plt.figure(figsize=(10, 6))
    sns.histplot(pixel_intensities, bins=64, kde=True)  # Reduced bins
    plt.title('Aggregated Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Folder path containing images
folder = 'plcae your path here'

# Load images in batches (using a generator)
batch_generator = load_images_from_folder(folder, batch_size=10)

# Aggregate pixel intensities from all images
pixel_intensities = np.concatenate([list(aggregate_pixel_intensities(batch)) for batch in batch_generator])

# Plot histogram
plot_histogram(pixel_intensities)

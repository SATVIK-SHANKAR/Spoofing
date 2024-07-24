import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path, target_size):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized_img = cv2.resize(img, target_size)
        return resized_img
    return None

def plot_heatmap(image1, image2, heatmap1, heatmap2, title1, title2):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 0].set_title(title1 + " - Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title(title2 + " - Original")
    axes[0, 1].axis('off')
    
    im1 = axes[1, 0].imshow(heatmap1, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title(title1 + " - Heatmap")
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(heatmap2, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title(title2 + " - Heatmap")
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

def process_and_plot(real_folder, spoof_folder, target_size):
    for filename in os.listdir(real_folder):
        real_path = os.path.join(real_folder, filename)
        spoof_path = os.path.join(spoof_folder, filename)
        
        if not os.path.exists(spoof_path):
            print(f"Skipping {filename}: Corresponding spoof image not found.")
            continue
        
        real_img = load_image(real_path, target_size)
        spoof_img = load_image(spoof_path, target_size)
        
        if real_img is None or spoof_img is None:
            print(f"Skipping {filename}: Failed to load one or both images.")
            continue
        
        # Generate heatmaps
        real_heatmap = real_img.astype(np.float64)
        spoof_heatmap = spoof_img.astype(np.float64)
        
        # Edge detection heatmaps
        real_edges = cv2.Canny(real_img, 100, 200)
        spoof_edges = cv2.Canny(spoof_img, 100, 200)
        
        # Plot heatmaps
        plot_heatmap(real_img, spoof_img, real_heatmap, spoof_heatmap, filename + " - Real", filename + " - Spoof")
        plot_heatmap(real_img, spoof_img, real_edges, spoof_edges, filename + " - Real Edges", filename + " - Spoof Edges")

# Paths to the folders containing real and spoof images
real_folder = 'plcae your path here'
spoof_folder = 'plcae your path here'

# Define the target size for resizing images (e.g., 224x224)
target_size = (224, 224)

# Process images and plot heatmaps
process_and_plot(real_folder, spoof_folder, target_size)

#! Based onPixel Intensity and Edge detection

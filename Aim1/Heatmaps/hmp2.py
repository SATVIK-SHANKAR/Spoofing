import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

def load_image(file_path, target_size):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized_img = cv2.resize(img, target_size)
        return resized_img
    return None

def compute_lbp(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    return lbp

def plot_heatmap(image1, image2, heatmaps, titles, main_title):
    num_heatmaps = len(heatmaps[0])
    fig, axes = plt.subplots(num_heatmaps, 2, figsize=(12, num_heatmaps * 6))
    
    if num_heatmaps == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, (heatmap1, heatmap2, title) in enumerate(zip(heatmaps[0], heatmaps[1], titles)):
        axes[i, 0].imshow(heatmap1, cmap='hot', interpolation='nearest')
        axes[i, 0].set_title(main_title + " - Real " + title)
        axes[i, 0].axis('off')
        fig.colorbar(axes[i, 0].imshow(heatmap1, cmap='hot', interpolation='nearest'), ax=axes[i, 0])
        
        axes[i, 1].imshow(heatmap2, cmap='hot', interpolation='nearest')
        axes[i, 1].set_title(main_title + " - Spoof " + title)
        axes[i, 1].axis('off')
        fig.colorbar(axes[i, 1].imshow(heatmap2, cmap='hot', interpolation='nearest'), ax=axes[i, 1])
    
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
        real_heatmap_intensity = real_img.astype(np.float64)
        spoof_heatmap_intensity = spoof_img.astype(np.float64)
        
        real_heatmap_edges = cv2.Canny(real_img, 100, 200)
        spoof_heatmap_edges = cv2.Canny(spoof_img, 100, 200)
        
        real_heatmap_lbp = compute_lbp(real_img)
        spoof_heatmap_lbp = compute_lbp(spoof_img)
        
        heatmaps_real = [real_heatmap_intensity, real_heatmap_edges, real_heatmap_lbp]
        heatmaps_spoof = [spoof_heatmap_intensity, spoof_heatmap_edges, spoof_heatmap_lbp]
        
        heatmaps = [heatmaps_real, heatmaps_spoof]
        titles = ["Intensity", "Edges", "LBP"]
        
        plot_heatmap(real_img, spoof_img, heatmaps, titles, filename)
        
# Paths to the folders containing real and spoof images
real_folder = 'plcae your path here'
spoof_folder = 'plcae your path here'


# Define the target size for resizing images (e.g., 224x224)
target_size = (224, 224)

# Process images and plot heatmaps
process_and_plot(real_folder, spoof_folder, target_size)


#! This is based on pixel intensity, edge detection, and Local Binary Patterns (LBP) for texture analysis.

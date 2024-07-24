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

def plot_heatmap(image1, image2, main_title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image1, cmap='hot', interpolation='nearest')
    axes[0].set_title(main_title + " - Real")
    axes[0].axis('off')
    fig.colorbar(axes[0].imshow(image1, cmap='hot', interpolation='nearest'), ax=axes[0])
    
    axes[1].imshow(image2, cmap='hot', interpolation='nearest')
    axes[1].set_title(main_title + " - Spoof")
    axes[1].axis('off')
    fig.colorbar(axes[1].imshow(image2, cmap='hot', interpolation='nearest'), ax=axes[1])
    
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
        
        plot_heatmap(real_img, spoof_img, filename)
        
# Paths to the folders containing real and spoof images
real_folder = 'plcae your path here'
spoof_folder = 'plcae your path here'

# Define the target size for resizing images (e.g., 224x224)
target_size = (224, 224)

# Process images and plot heatmaps
process_and_plot(real_folder, spoof_folder, target_size)



#! Based on Simple pixel intensity
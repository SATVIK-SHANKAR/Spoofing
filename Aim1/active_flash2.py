import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return images

def apply_lighting_conditions(image, alpha_values):
    images_with_lighting = []
    for alpha in alpha_values:
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        images_with_lighting.append(new_image)
    return images_with_lighting

def plot_images(images, title):
    n = len(images)
    plt.figure(figsize=(20, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.show()

def compute_and_plot_differences(original, altered_images):
    n = len(altered_images)
    plt.figure(figsize=(20, 10))
    for i in range(n):
        diff = cv2.absdiff(original, altered_images[i])
        plt.subplot(3, n, i + 1)
        plt.imshow(cv2.cvtColor(altered_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Altered {i+1}')
        plt.axis('off')
        
        plt.subplot(3, n, n + i + 1)
        plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        plt.title(f'Difference {i+1}')
        plt.axis('off')
        
        plt.subplot(3, n, 2 * n + i + 1)
        hist_diff = cv2.calcHist([diff], [0], None, [256], [0, 256])
        plt.plot(hist_diff)
        plt.title(f'Histogram Diff {i+1}')
        plt.axis('off')
    plt.show()

folder = "path to folder"
images = load_images_from_folder(folder)

# Simulate lighting conditions
alpha_values = [0.5, 1.0, 1.5, 2.0]  # Different brightness levels

# Apply lighting conditions to each image and plot the results
for img in images:
    images_with_lighting = apply_lighting_conditions(img, alpha_values)
    compute_and_plot_differences(img, images_with_lighting)

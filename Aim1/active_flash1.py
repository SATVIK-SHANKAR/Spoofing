import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_image_pairs(folder):
    images_before_flash = []
    images_after_flash = []
    for filename in sorted(os.listdir(folder)):
        if 'before' in filename:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images_before_flash.append(img)
        elif 'after' in filename:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images_after_flash.append(img)
    return images_before_flash, images_after_flash

def plot_image_pairs(before_images, after_images):
    n = len(before_images)
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(cv2.cvtColor(before_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Before Flash {i+1}')
        plt.axis('off')
        
        plt.subplot(3, n, n + i + 1)
        plt.imshow(cv2.cvtColor(after_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'After Flash {i+1}')
        plt.axis('off')
        
        diff = cv2.absdiff(before_images[i], after_images[i])
        plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        plt.title(f'Difference {i+1}')
        plt.axis('off')
    plt.show()

folder = "path to folder"
before_images, after_images = load_image_pairs(folder)
plot_image_pairs(before_images, after_images)

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))  # Sort to ensure matching order
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, filenames

def compute_keypoints_and_descriptors(images):
    orb = cv2.ORB_create()  # Use ORB instead of SURF
    keypoints_and_descriptors = []
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        keypoints_and_descriptors.append((keypoints, descriptors))
    return keypoints_and_descriptors

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def differentiate_images(folder1, folder2):
    images1, filenames1 = load_images_from_folder(folder1)
    images2, filenames2 = load_images_from_folder(folder2)
    
    kp_and_desc1 = compute_keypoints_and_descriptors(images1)
    kp_and_desc2 = compute_keypoints_and_descriptors(images2)
    
    good_matches_counts = []
    for (kp1, desc1), (kp2, desc2) in zip(kp_and_desc1, kp_and_desc2):
        if desc1 is not None and desc2 is not None:
            good_matches = match_descriptors(desc1, desc2)
            good_matches_counts.append(len(good_matches))
        else:
            good_matches_counts.append(0)
    
    return good_matches_counts, filenames1

def plot_results(matches_counts, filenames):
    plt.figure(figsize=(12, 6))
    plt.plot(matches_counts, 'bo-', label='Good Matches Count')
    plt.title('Good Matches Count Between Original and Spoof Images')
    plt.xlabel('Image Index')
    plt.ylabel('Number of Good Matches')
    plt.xticks(ticks=range(len(filenames)), labels=filenames, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Folder paths
folder1 = 'plcae your path here'  # Folder containing original images
folder2 = 'place your path here'  # Folder containing spoof images

# Differentiate images and get good matches counts
matches_counts, filenames = differentiate_images(folder1, folder2)

# Plot the results
plot_results(matches_counts, filenames)

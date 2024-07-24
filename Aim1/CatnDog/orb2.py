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

def differentiate_images(images):
    num_images = len(images)
    matches_counts = []
    
    for i in range(num_images):
        img = images[i]
        kp_and_desc = compute_keypoints_and_descriptors([img])
        
        if len(kp_and_desc) > 0:
            kp, desc = kp_and_desc[0]
            good_matches_count = len(match_descriptors(desc, desc))  # Use the same image to match with itself
        else:
            good_matches_count = 0
        
        matches_counts.append(good_matches_count)
    
    return matches_counts

def plot_results(matches_counts):
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(matches_counts)), matches_counts, color='blue')
    plt.title('Good Matches Count Between Original and Spoof Images')
    plt.xlabel('Image Index')
    plt.ylabel('Number of Good Matches')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Folder path containing both original and spoof images
folder = 'plcae your path here'

# Load images from folder
images, _ = load_images_from_folder('plcae your path here')

# Differentiate images and get good matches counts
matches_counts = differentiate_images(images)

# Plot the results
plot_results(matches_counts)

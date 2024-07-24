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
    surf = cv2.xfeatures2d.SURF_create(400)  # Hessian Threshold
    keypoints_and_descriptors = []
    for img in images:
        keypoints, descriptors = surf.detectAndCompute(img, None)
        keypoints_and_descriptors.append((keypoints, descriptors))
    return keypoints_and_descriptors

def match_descriptors(desc1, desc2):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

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

def plot_results(matches_counts, filenames, num_original, num_spoof):
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(num_original), matches_counts[:num_original], color='blue', label='Original Images')
    plt.bar(np.arange(num_original, num_original + num_spoof), matches_counts[num_original:], color='orange', label='Spoof Images')
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
folder2 = 'plcae your path here'  # Folder containing spoof images


# Differentiate images and get good matches counts
matches_counts, filenames = differentiate_images(folder1, folder2)

# Plot the results
plot_results(matches_counts, filenames,40,40)

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the folder containing your images
image_folder = 'plcae your path here'

# Get a list of image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Initialize an empty list to store histograms
histograms = []

# Loop through each image file
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Compute the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Append the histogram to the list
    histograms.append(hist)

# Combine histograms
combined_hist = np.sum(histograms, axis=0)

# Plot the combined histogram
plt.figure(figsize=(10, 6))
plt.plot(combined_hist, color='b')
plt.title('Originial')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the folder containing your images
image_folder = 'plcae your path here'

# Get a list of image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Initialize an empty list to store histograms
histograms = []

# Loop through each image file
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Compute the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Append the histogram to the list
    histograms.append(hist)

# Combine histograms
combined_hist = np.sum(histograms, axis=0)

# Plot the combined histogram
plt.figure(figsize=(10, 6))
plt.plot(combined_hist, color='b')
plt.title('Spoof images')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

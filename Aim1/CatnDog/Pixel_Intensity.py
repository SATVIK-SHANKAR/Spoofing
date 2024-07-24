import cv2
import os
from PIL import Image, ImageFilter
import numpy as np

def generate_spoof_images(input_dir, output_dir, num_images=40):
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        if i < 10:
            create_print_attack(image, output_dir, i)
        elif i < 20:
            create_display_attack(image, output_dir, i - 10)
        elif i < 30:
            create_blur_attack(image, output_dir, i - 20)
        elif i < 40:
            create_distortion_attack(image, output_dir, i - 30)

def create_print_attack(image, output_path, idx):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noisy_image = add_noise(gray_image)
    print_image = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2RGB)
    output_filename = os.path.join(output_path, f'print_attack_{idx}.jpg')
    cv2.imwrite(output_filename, print_image)

def add_noise(image):
    row, col = image.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def create_display_attack(image, output_path, idx):
    rows, cols, _ = image.shape
    reflection = np.zeros_like(image)
    cv2.rectangle(reflection, (int(cols*0.3), int(rows*0.3)), (int(cols*0.7), int(rows*0.7)), (255, 255, 255), -1)
    display_image = cv2.addWeighted(image, 0.8, reflection, 0.2, 0)
    output_filename = os.path.join(output_path, f'display_attack_{idx}.jpg')
    cv2.imwrite(output_filename, display_image)

def create_blur_attack(image, output_path, idx):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(5))
    blurred_image = cv2.cvtColor(np.array(blurred_image), cv2.COLOR_RGB2BGR)
    output_filename = os.path.join(output_path, f'blur_attack_{idx}.jpg')
    cv2.imwrite(output_filename, blurred_image)

def create_distortion_attack(image, output_path, idx):
    rows, cols, _ = image.shape
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([[0, 0], [cols*0.8, rows*0.1], [cols*0.2, rows*0.9], [cols-1, rows-1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    distorted_image = cv2.warpPerspective(image, matrix, (cols, rows))
    output_filename = os.path.join(output_path, f'distortion_attack_{idx}.jpg')
    cv2.imwrite(output_filename, distorted_image)

input_dir = 'plcae your path here'
output_dir = 'plcae your path here'
os.makedirs(output_dir, exist_ok=True)
generate_spoof_images(input_dir, output_dir)


# **********************************************************************************************
#& Feature exttraction by the below method does not provide very useful differentiating results as below laterwe shall use better methods

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (224, 224))
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)
    return mean_intensity, std_intensity

def create_dataset(genuine_dir, spoof_dir):
    features = []
    labels = []

    # Extract features from genuine images
    for image_file in os.listdir(genuine_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(genuine_dir, image_file)
            mean_intensity, std_intensity = extract_features(image_path)
            features.append([mean_intensity, std_intensity])
            labels.append('genuine')

    # Extract features from spoof images
    for image_file in os.listdir(spoof_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(spoof_dir, image_file)
            mean_intensity, std_intensity = extract_features(image_path)
            features.append([mean_intensity, std_intensity])
            labels.append('spoof')

    return np.array(features), np.array(labels)

def plot_dataset(features, labels):
    genuine_features = features[labels == 'genuine']
    spoof_features = features[labels == 'spoof']

    plt.figure(figsize=(10, 6))
    plt.scatter(genuine_features[:, 0], genuine_features[:, 1], label='Genuine', alpha=0.5)
    plt.scatter(spoof_features[:, 0], spoof_features[:, 1], label='Spoof', alpha=0.5)
    plt.xlabel('Mean Intensity')
    plt.ylabel('Standard Deviation of Intensity')
    plt.legend()
    plt.title('Genuine vs Spoof Images')
    plt.show()

genuine_dir = 'plcae your path here'
spoof_dir = 'plcae your path here'
features, labels = create_dataset(genuine_dir, spoof_dir)
plot_dataset(features, labels)


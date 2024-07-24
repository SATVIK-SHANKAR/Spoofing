import cv2
import numpy as np

def remove_background(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()

    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

    # Create an initial mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create temporary arrays used by the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to make sure the background is set to 0 and the foreground is set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Extract the foreground using the mask
    foreground = image * mask2[:, :, np.newaxis]

    # Display the results
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Foreground', foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_path = 'plcae ur image path'
# Remove the background
# cv2.imwrite(output_path, foreground)
remove_background(image_path)

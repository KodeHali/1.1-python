import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def kmeans_clustering(image_path, result_folder, K=3):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers back to 8-bit values
    centers = np.uint8(centers)

    # Map labels to center colors
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Save the segmented image in the result folder
    result_image_path = os.path.join(result_folder, 'result_kmeans_' + os.path.basename(image_path))
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

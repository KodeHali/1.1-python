import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

def meanshift_clustering(image_path, result_folder):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to make Mean Shift more efficient
    max_dimension = 250  # Set the maximum dimension size
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / float(max(height, width))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    # Reshape the image to a 2D array of pixels
    pixel_values = resized_image.reshape((-1, 3))

    # Apply Mean Shift Clustering
    meanshift = MeanShift(bin_seeding=True)
    print("Applying Mean Shift clustering...")
    meanshift.fit(pixel_values)

    # Extract labels and unique centers (colors)
    labels = meanshift.labels_
    centers = meanshift.cluster_centers_

    # Convert centers back to 8-bit values
    centers = np.uint8(centers)

    # Map labels to center colors
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(resized_image.shape)

    # Save the segmented image
    result_image_path = os.path.join(result_folder, 'result_meanshift_' + os.path.basename(image_path))
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

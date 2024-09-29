import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

def meanshift_clustering(image_path, result_folder):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to make Mean Shift more efficient
    scale_percent = 50  # Reduce image size to 50% of the original
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Reshape the image to a 2D array of pixels
    pixel_values = resized_image.reshape((-1, 3))

    # Apply Mean Shift Clustering
    meanshift = MeanShift(bin_seeding=True)
    meanshift.fit(pixel_values)

    # Extract labels and unique centers (colors)
    labels = meanshift.labels_
    centers = meanshift.cluster_centers_

    # Convert centers back to 8-bit values
    centers = np.uint8(centers)

    # Map labels to center colors
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(resized_image.shape)

    # Save the segmented image in the static/uploads/ folder
    result_image_path = os.path.join(result_folder, 'result_meanshift_' + os.path.basename(image_path))
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

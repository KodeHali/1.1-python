import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

def birch_clustering(image_path, result_folder, n_clusters=5, threshold=0.5, branching_factor=50, max_dimension=500):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original dimensions
    original_height, original_width = image.shape[:2]

    # Resize the image to make BIRCH more efficient
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / float(max(height, width))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        print(f"Image resized to {new_size[1]}x{new_size[0]} for processing.")
    else:
        resized_image = image
        print("Image size is within the limit; no resizing applied.")

    # Reshape the image to a 2D array of pixels
    pixel_values = resized_image.reshape((-1, 3))
    pixel_values = pixel_values / 255.0  # Normalize color values to [0, 1]

    # Get spatial coordinates
    h, w = resized_image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coordinates = np.stack((yy, xx), axis=2).reshape(-1, 2)
    coordinates = coordinates / np.max(coordinates)  # Normalize coordinates to [0, 1]

    # Combine color and spatial information into a feature vector
    feature_vector = np.concatenate((pixel_values, coordinates), axis=1)

    # Standardize the feature vector
    scaler = StandardScaler()
    feature_vector = scaler.fit_transform(feature_vector)

    # Apply BIRCH Clustering
    print("Applying BIRCH clustering...")
    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    birch.fit(feature_vector)

    labels = birch.labels_
    unique_labels = np.unique(labels)

    # Handle the case where all points are assigned to one cluster
    if len(unique_labels) == 1:
        print("BIRCH found only one cluster.")
        # Assign the mean color of the entire image
        segmented_image = np.tile(np.mean(pixel_values, axis=0), (pixel_values.shape[0], 1))
    else:
        # Calculate the mean color of each cluster
        segmented_image = np.zeros_like(pixel_values)
        for label in unique_labels:
            cluster_mean = np.mean(pixel_values[labels == label], axis=0)
            segmented_image[labels == label] = cluster_mean

    # Reshape back to image shape and scale back to [0, 255]
    segmented_image = segmented_image.reshape(resized_image.shape)
    segmented_image = (segmented_image * 255).astype(np.uint8)

    # Resize the segmented image back to the original dimensions
    segmented_image = cv2.resize(segmented_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Save the segmented image
    result_image_path = os.path.join(result_folder, 'result_birch_' + os.path.basename(image_path))
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

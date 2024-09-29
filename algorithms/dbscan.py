import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan_clustering(image_path, result_folder, eps=5, min_samples=5, max_dimension=300):

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original dimensions
    original_height, original_width = image.shape[:2]

    # Resize the image to make DBSCAN more efficient
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

    # Apply DBSCAN Clustering
    print("Applying DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    dbscan.fit(feature_vector)

    labels = dbscan.labels_
    unique_labels = np.unique(labels)

    # Handle the case where all points are considered noise
    if len(unique_labels) <= 1 and unique_labels[0] == -1:
        print("DBSCAN found no clusters.")
        # Create a black image to represent no clusters found
        segmented_image = np.zeros_like(resized_image)
    else:
        # Map labels to colors
        # Exclude noise label (-1) from the set of labels
        unique_labels = unique_labels[unique_labels >= 0]
        # Generate random colors for the clusters
        colors = [np.random.randint(0, 255, size=3) for _ in unique_labels]
        colors = np.array(colors, dtype='uint8')
        # Create a mapping from labels to colors
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        # Assign colors to each pixel
        segmented_image = np.zeros_like(resized_image)
        for idx, label in enumerate(labels):
            if label == -1:
                # Assign black color to noise points
                segmented_image[idx // w, idx % w] = [0, 0, 0]
            else:
                segmented_image[idx // w, idx % w] = label_to_color[label]

    # Resize the segmented image back to the original dimensions
    segmented_image = cv2.resize(segmented_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Save the segmented image
    result_image_path = os.path.join(result_folder, 'result_dbscan_' + os.path.basename(image_path))
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

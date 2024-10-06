import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan_clustering(feature_vector, original_shape, scaler, result_folder, use_pca=False, pca=None):
    # Use the constants
    eps = 0.05
    min_samples = 5
    max_dimension = 300
    # original_shape is the shape of the original image
    original_height, original_width = original_shape[:2]

    # Determine the scaling factor
    max_original_dimension = max(original_height, original_width)
    if max_original_dimension > max_dimension:
        scaling_factor = max_dimension / float(max_original_dimension)
        new_height = int(original_height * scaling_factor)
        new_width = int(original_width * scaling_factor)
        print(f"Resizing image from ({original_height}, {original_width}) to ({new_height}, {new_width})")
    else:
        scaling_factor = 1.0
        new_height = original_height
        new_width = original_width
        print("Image size is within the limit; no resizing applied.")

    # Reshape the feature vector back to image dimensions to handle resizing
    # This is necessary because we need to resize the image before clustering
    # and adjust the feature vector accordingly
    feature_vector_image = feature_vector.reshape((original_height, original_width, -1))

    # Resize the feature vector image
    # Initialize the resized feature vector image
    num_channels = feature_vector_image.shape[2]
    resized_feature_vector_image = np.zeros((new_height, new_width, num_channels), dtype=feature_vector_image.dtype)

    # Resize each channel individually
    for i in range(num_channels):
        channel = feature_vector_image[:, :, i]
        resized_channel = cv2.resize(channel, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_feature_vector_image[:, :, i] = resized_channel

    # Flatten the resized feature vector image back to (num_pixels, features)
    resized_feature_vector = resized_feature_vector_image.reshape((-1, feature_vector.shape[1]))

    # Now apply DBSCAN clustering on the resized feature vector
    print("Applying DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(resized_feature_vector)
    unique_labels = np.unique(labels)

    # Handle the case where all points are considered noise
    if len(unique_labels) <= 1 and unique_labels[0] == -1:
        print("DBSCAN found no clusters.")
        # Create a black image to represent no clusters found
        segmented_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    else:
        if use_pca:
            # Since PCA was applied and we cannot inverse transform, assign random colors to clusters
            unique_labels = unique_labels[unique_labels >= 0]
            colors = [np.random.randint(0, 255, size=3) for _ in unique_labels]
            colors = np.array(colors, dtype='uint8')
            label_to_color = {label: color for label, color in zip(unique_labels, colors)}
            # Assign colors to each pixel
            segmented_image = np.zeros((new_height * new_width, 3), dtype=np.uint8)
            for idx, label in enumerate(labels):
                if label == -1:
                    # Assign black color to noise points
                    segmented_image[idx] = [0, 0, 0]
                else:
                    segmented_image[idx] = label_to_color[label]
            segmented_image = segmented_image.reshape((new_height, new_width, 3))
        else:
            # When PCA is not used, we can inverse transform to get original pixel values
            non_noise_indices = np.where(labels != -1)[0]
            feature_vector_original = scaler.inverse_transform(resized_feature_vector)
            original_pixels = (feature_vector_original[:, :3] * 255).astype(np.uint8)
            segmented_image = np.zeros((new_height * new_width, 3), dtype=np.uint8)
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise
                cluster_indices = np.where(labels == label)[0]
                mean_color = np.mean(original_pixels[cluster_indices], axis=0)
                segmented_image[cluster_indices] = mean_color.astype(np.uint8)
            # Assign black to noise pixels
            noise_indices = np.where(labels == -1)[0]
            segmented_image[noise_indices] = [0, 0, 0]
            segmented_image = segmented_image.reshape((new_height, new_width, 3))

    # Resize the segmented image back to original dimensions
    segmented_image_full_size = cv2.resize(segmented_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Save the result
    result_image_path = os.path.join(result_folder, 'result_dbscan_' + os.path.basename('result.png'))
    plt.imsave(result_image_path, segmented_image_full_size)

    return result_image_path

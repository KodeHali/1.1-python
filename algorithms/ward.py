import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

def ward_clustering(image, scaler, result_folder, use_pca=False, pca=None, n_clusters=5):

    # Resize the image if necessary
    max_dimension = 200  # Adjust as needed
    original_height, original_width = image.shape[:2]
    if max(original_height, original_width) > max_dimension:
        scaling_factor = max_dimension / max(original_height, original_width)
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Image resized to {new_width}x{new_height}")
    else:
        print("Image resizing not required.")

    # Update original_shape after resizing
    original_shape = image.shape

    # Prepare the feature vector
    pixel_values = image.reshape((-1, 3)) / 255.0  # Normalize pixel values

    # Get spatial coordinates
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coordinates = np.stack((yy, xx), axis=2).reshape(-1, 2)
    coordinates = coordinates / np.max(coordinates)  # Normalize coordinates

    # Combine color and spatial information into a feature vector
    feature_vector = np.concatenate((pixel_values, coordinates), axis=1)

    # Standardize the feature vector
    feature_vector_scaled = scaler.fit_transform(feature_vector)

    # Apply PCA if selected
    if use_pca and pca is not None:
        print("Applying PCA...")
        feature_vector_transformed = pca.fit_transform(feature_vector_scaled)
    else:
        feature_vector_transformed = feature_vector_scaled

    # Apply Ward's method using Agglomerative Clustering
    print("Applying Ward's method clustering...")
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = ward.fit_predict(feature_vector_transformed)

    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels)

    # Initialize an array to hold the segmented image
    segmented_image = np.zeros((labels.shape[0], 3))

    if use_pca and pca is not None:
        # PCA was used
        cluster_centers_pca = np.zeros((n_clusters_found, feature_vector_transformed.shape[1]))
        for i, label in enumerate(unique_labels):
            cluster_mean = feature_vector_transformed[labels == label].mean(axis=0)
            cluster_centers_pca[i] = cluster_mean

        # Inverse transform back to original feature space
        cluster_centers_original = scaler.inverse_transform(pca.inverse_transform(cluster_centers_pca))

        # Assign colors to pixels based on cluster centers
        for i, label in enumerate(unique_labels):
            segmented_image[labels == label] = cluster_centers_original[i, :3]
    else:
        # PCA was not used
        cluster_centers_scaled = np.zeros((n_clusters_found, feature_vector_scaled.shape[1]))
        for i, label in enumerate(unique_labels):
            cluster_mean = feature_vector_scaled[labels == label].mean(axis=0)
            cluster_centers_scaled[i] = cluster_mean

        # Inverse transform to get back to original feature space
        cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

        # Assign colors to pixels based on cluster centers
        for i, label in enumerate(unique_labels):
            segmented_image[labels == label] = cluster_centers_original[i, :3]

    # Scale pixel values back to [0, 255] and convert to uint8
    segmented_image = (segmented_image * 255).astype(np.uint8)
    segmented_image = segmented_image.reshape(original_shape)

    # Save the result
    if use_pca:
        result_image_path = os.path.join(result_folder, 'result_ward_pca.png')
    else:
        result_image_path = os.path.join(result_folder, 'result_ward.png')
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

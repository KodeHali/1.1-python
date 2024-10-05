import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def kmeans_clustering(feature_vector, original_shape, scaler, result_folder, use_pca=False, K=5):

    # Apply K-Means Clustering
    print("Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(feature_vector)
    centers = kmeans.cluster_centers_

    # Reconstruct the segmented image
    if use_pca:
        # Since PCA was applied, we cannot inverse transform using scaler
        # Assign colors based on cluster labels or use the inverse PCA transform if possible
        segmented_image = centers[labels]
        # Optionally, you can attempt to inverse transform using PCA if you've stored it
        # For simplicity, we'll map each cluster to its center's color components
        segmented_image = (segmented_image * 255).astype(np.uint8)
    else:
        # Inverse transform to get back to original feature space
        segmented_image = scaler.inverse_transform(centers)[labels][:, :3]
        segmented_image = (segmented_image * 255).astype(np.uint8)

    segmented_image = segmented_image.reshape(original_shape)

    # Save the result
    result_image_path = os.path.join(result_folder, 'result_kmeans.png')
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

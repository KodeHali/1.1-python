import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

def birch_clustering(feature_vector, original_shape, scaler, result_folder, use_pca=False, pca=None, n_clusters=5, threshold=0.5, branching_factor=50):
    # Apply BIRCH Clustering
    print("Applying BIRCH clustering...")
    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    labels = birch.fit_predict(feature_vector)
    unique_labels = np.unique(labels)

    # Reconstruct the segmented image
    if use_pca and pca is not None:
        # Inverse transform using PCA and scaler
        centers = birch.subcluster_centers_
        centers_inverse_pca = pca.inverse_transform(centers)
        centers_original = scaler.inverse_transform(centers_inverse_pca)
    else:
        # Inverse transform to get back to original feature space
        centers = birch.subcluster_centers_
        centers_original = scaler.inverse_transform(centers)

    # Map labels to colors
    segmented_image = centers_original[labels][:, :3]
    segmented_image = (segmented_image * 255).astype(np.uint8)
    segmented_image = segmented_image.reshape(original_shape)

    # Save the result
    if use_pca:
        result_image_path = os.path.join(result_folder, 'result_birch_pca.png')
    else:
        result_image_path = os.path.join(result_folder, 'result_birch.png')
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

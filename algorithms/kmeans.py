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

    print(f"Cluster centers for K={K}:\n{centers}")


    if use_pca:
        segmented_image = centers[labels]
        segmented_image = (segmented_image * 255).astype(np.uint8)
    else:
        segmented_image = scaler.inverse_transform(centers)[labels][:, :3]
        segmented_image = (segmented_image * 255).astype(np.uint8)

    segmented_image = segmented_image.reshape(original_shape)
    if use_pca:
        result_image_path = os.path.join(result_folder, 'result_kmeans_pca.png')
    else:
        result_image_path = os.path.join(result_folder, 'result_kmeans.png')
    plt.imsave(result_image_path, segmented_image)

    return result_image_path

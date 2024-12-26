from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

csv_files = []

def objective(eps):
    file = csv_files[1]
    pivot_table = pd.read_csv(file)
    # Prepare the feature matrix
    y1=pivot_table['xc1']
    y2=pivot_table['xc2']
    x=np.linspace(0,1600,y1.size)

    # Combine y1 and y2 into a single array for clustering
    y_combined = np.concatenate((y1, y2))
    x_combined = np.concatenate((x, x))

    # Create a feature matrix
    features = np.column_stack((x_combined, y_combined))

    # Number of neighbors
    k = 50  # Typically min_samples

        # Standardize the feature matrix
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)


# Compute the k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(scaled_features)
    distances, indices = neighbors_fit.kneighbors(scaled_features)

    # Sort and plot the distances
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.ylabel(f'Distance to {k}th Nearest Neighbor')
    plt.xlabel('Data Points sorted by distance')
    plt.title('k-Distance Graph to Determine eps')
    plt.grid(True)
    plt.show()  

    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=50)  # Adjust eps and min_samples as needed

    # Fit the model
    cluster_labels = dbscan.fit_predict(features)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    unique_labels = set(cluster_labels)

    for label in unique_labels:
        # Select points belonging to the current cluster
        indices = cluster_labels == label
        plt.scatter(features[indices, 0], features[indices, 1], label=f'Cluster {label}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'DBSCAN Clustering eps={eps}')
    plt.legend()
    plt.show()

# List of CSV file paths
for roots, dirs, files in os.walk("data\\yig_t_sweep_outputs\\peaks_widths"):
    csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]# if dirs == ['intermediaries', 'peaks_widths']]

# for file in csv_files:
if __name__ == "__main__":
    # Load the data
    # for i in range(1, 10):
        # objective(i)
    objective(630)
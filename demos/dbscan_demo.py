import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

# Your own library imports
from GrimoireML.grimoireml.clusters.dbscan import DBSCAN

def fetch_data() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch Iris data and return the feature matrix and labels."""
    data = load_iris()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def run_dbscan(X: np.ndarray, eps: float = .55, min_points: int = 5) -> np.ndarray:
    """Run DBSCAN clustering algorithm and return cluster labels."""
    dbscan = DBSCAN(eps=eps, min_points=min_points)
    return dbscan.fit(X)

def plot_3d(X_real: np.ndarray, y_real: np.ndarray, X_clustered: np.ndarray, labels: np.ndarray):
    """Plot 3D scatter plots side by side."""
    fig = plt.figure(figsize=(12, 6))

    # Plot real data
    ax1 = fig.add_subplot(121, projection='3d')
    unique_labels_real = np.unique(y_real)
    colors_real = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels_real)))
    for label, color in zip(unique_labels_real, colors_real):
        points = X_real[y_real == label]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], label=f'Class {label}', alpha=0.6, edgecolors='k')
    ax1.set_title('Real Data')
    ax1.legend()

    # Plot clustered data
    ax2 = fig.add_subplot(122, projection='3d')
    unique_labels_clustered = np.unique(labels)
    colors_clustered = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels_clustered)))
    for label, color in zip(unique_labels_clustered, colors_clustered):
        points = X_clustered[labels == label]
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], label=f'Cluster {label}', alpha=0.6, edgecolors='k')
    ax2.set_title('DBSCAN Clustered Data')
    ax2.legend()

    plt.show()

def main():
    # Fetch and preprocess data
    X, y = fetch_data()

    # Run DBSCAN
    start_time = timer()
    labels = run_dbscan(X)
    end_time = timer()
    print(f"Time taken for DBSCAN: {end_time - start_time} seconds")

    # Reduce dimensionality for visualization
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    # Plot both real and clustered data side by side
    plot_3d(X_reduced, y, X_reduced, labels)

if __name__ == "__main__":
    main()

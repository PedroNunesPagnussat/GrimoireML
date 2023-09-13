import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.datasets import load_iris, make_blobs, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Union

# Your own library imports
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# (Assuming your library is in the Python path or the same directory)
from grimoireml.clusters.dbscan import DBSCAN

def fetch_data(data: str = 'blob') -> Tuple[np.ndarray, np.ndarray]:
    """Fetch data and return the feature matrix and labels."""
    if data == 'blob':
        X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)
    elif data == 'iris':
        iris = load_iris()
        X, y = iris.data, iris.target
    elif data == 'wine':
        wine = load_wine()
        X, y = wine.data, wine.target
    else:
        raise ValueError("Invalid data source")

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def run_dbscan(X: np.ndarray, eps: float = .55, min_points: int = 5) -> np.ndarray:
    """Run custom DBSCAN clustering algorithm and return cluster labels."""
    dbscan = DBSCAN(eps=eps, min_points=min_points)
    return dbscan.fit(X)

def run_sklearn_dbscan(X: np.ndarray, eps: float = .55, min_samples: int = 5) -> np.ndarray:
    """Run scikit-learn's DBSCAN clustering algorithm and return cluster labels."""
    dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    return dbscan.labels_


def plot_3d(X: np.ndarray, labels: np.ndarray, title: str):
    """Plot 3D scatter plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        points = X[labels == label]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], label=f'Cluster {label}', alpha=0.6, edgecolors='k')
    ax.set_title(title)
    ax.legend()
    plt.show()

from sklearn.metrics import adjusted_rand_score

def main():
    # Fetch and preprocess data
    X, y = fetch_data('blob')  # You can change this to 'blob' or 'wine'

    # Run custom DBSCAN
    start_time = timer()
    custom_labels = run_dbscan(X)
    end_time = timer()
    custom_time = end_time - start_time
    print(f"Time taken for custom DBSCAN: {custom_time} seconds")

    # Run scikit-learn DBSCAN
    start_time = timer()
    sklearn_labels = run_sklearn_dbscan(X)
    end_time = timer()
    sklearn_time = end_time - start_time
    print(f"Time taken for scikit-learn DBSCAN: {sklearn_time} seconds")

    # Calculate similarity between custom and scikit-learn DBSCAN using ARI
    similarity = np.sum(custom_labels == sklearn_labels) / len(custom_labels) * 100
    similarity = str(np.round(similarity, 2)) + "%"
    print(f"Adjusted Rand Index between custom and scikit-learn DBSCAN: {similarity}")


        # Reduce dimensionality for visualization
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    # Plot both custom and scikit-learn DBSCAN results
    plot_3d (X_reduced, y, "Real Data")
    plot_3d(X_reduced, custom_labels, 'Custom DBSCAN')
    plot_3d(X_reduced, sklearn_labels, 'Scikit-learn DBSCAN')


if __name__ == "__main__":
    main()
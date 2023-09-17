import sys
import os


current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(project_root_directory)
from grimoireml.Cluster import DBSCAN  # noqa: E402
from grimoireml.Functions.EvaluationFunctions import Accuracy  # noqa: E402


def main():
    # fetch iris dataset from sklearn
    from sklearn import datasets

    iris = datasets.load_iris()

    X, y = iris.data, iris.target

    epsilon = 0.55
    min_points = 5
    db = DBSCAN(epsilon=epsilon, min_points=min_points)
    clusters = db.fit(X)  # This will give you the clusters
    clusters = db.clusters  # This will give you the clusters as well

    accuracy = Accuracy("multiclass")
    accuracy(clusters, y)  # This will give you the accuracy of the model


if __name__ == "__main__":
    main()

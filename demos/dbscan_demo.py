import numpy as np
import sys
import os

current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
sys.path.append(project_root_directory)


from grimoireml.Cluster import DBSCAN



def main():
    # fetch iris dataset from sklearn
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data, iris.target   


    eplison = 0.55
    min_points = 5
    db = DBSCAN(epsilon=eplison, min_points=min_points)
    db.fit(X)
    clusters = db.clusers

    # You can also do this
    # cluster = db.fit(X)


if __name__ == "__main__":
    main()
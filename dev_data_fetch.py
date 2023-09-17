from sklearn import datasets


def iris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    return X, y


def wine():
    wine = datasets.load_wine()
    X, y = wine.data, wine.target

    return X, y


def breast_cancer():
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    return X, y


def mnist():
    mnist = datasets.load_digits()
    X, y = mnist.data, mnist.target

    return X, y


def fetch_data(dataset: str):
    if dataset == "iris":
        return iris()
    elif dataset == "wine":
        return wine()
    elif dataset == "breast_cancer":
        return breast_cancer()
    elif dataset == "mnist":
        return mnist()
    else:
        raise ValueError("Dataset not found")

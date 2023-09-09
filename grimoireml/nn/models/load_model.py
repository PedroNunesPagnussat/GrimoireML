from .sequential import Sequential
import pickle

def load_model(path: str) -> Sequential:
    """Load a model from a file.

    Args:
        path: The path to the file.
    """

    with open(path, "rb") as f:
        loaded_dict = pickle.load(f)

    return loaded_dict
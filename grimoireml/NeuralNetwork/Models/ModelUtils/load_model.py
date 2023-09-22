import pickle


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
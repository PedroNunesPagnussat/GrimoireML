import pickle


def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

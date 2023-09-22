import pickle


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
<<<<<<< HEAD
    return model
=======
    return model
>>>>>>> 2d3842c4d58575ad6709c8f435b2ef0b01b39569

import numpy as np
from timeit import default_timer as timer
from typing import Tuple

# Your own library imports
from GrimoireML.grimoireml.nn.sequential import Sequential

from GrimoireML.grimoireml.nn.layers import Dense
from GrimoireML.grimoireml.nn.functions.activation_functions import Sigmoid, ReLU
from GrimoireML.grimoireml.nn.optimizers import SGD
from GrimoireML.grimoireml.nn.functions.loss_functions import BCE

import GrimoireML.grimoireml.functions.evaluation_functions as eval_functions






def fatch_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steal data from sklearn and return train/test splits."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape: int) -> Sequential:
    """Build and return the model."""
    model = Sequential()
    model.add(Dense(input_shape, 16, activation=Sigmoid()))
    model.add(Dense(16, 8, activation=ReLU()))
    model.add(Dense(8, 4, activation=ReLU()))
    model.add(Dense(4, 2, activation=ReLU()))
    model.add(Dense(2, 1, activation=Sigmoid()))
    model.compile(optimizer=SGD(lr=0.025), loss=BCE(), metrics=[eval_functions.MSE()])
    return model


def main():
    # set random seed
    np.random.seed(42)
    
    X_train, X_test, y_train, y_test = fatch_data()
    epochs = 20
    batch_size = 16
    input_shape = X_train.shape[1]
    y_train = y_train.reshape(-1, 1)

    print("GML")

    time_start = timer()
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # predict and evaluate
    pred = model.predict(X_test[0])
    time_end = timer()
    total_time = time_end - time_start
    
    print(f"GML Time: {round(total_time, 4)}")
    print(pred, y_test[0])


if __name__ == "__main__":
    main()
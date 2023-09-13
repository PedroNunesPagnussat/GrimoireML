import numpy as np
from timeit import default_timer as timer
from typing import Tuple
import sys
import os.path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Your own library imports
from grimoireml.nn.models.sequential import Sequential

from grimoireml.nn.layers import Dense
from grimoireml.nn.functions.activation_functions import Sigmoid, ReLU
from grimoireml.nn.optimizers import SGD, Adagrad, Adam
from grimoireml.nn.functions.loss_functions import BCE, CCE

import grimoireml.functions.evaluation_functions as eval_functions






def fetch_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steal data from sklearn and return train/test splits."""

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
    model.compile(optimizer=Adam(lr=0.05), loss=CCE(), metrics=[eval_functions.MSE()])
    return model


def main():
    # set random seed
    np.random.seed(42)
    
    X_train, X_test, y_train, y_test = fetch_data()
    
    # Fetch some validation data from X_train and y_train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    epochs = 20
    batch_size = 36
    input_shape = X_train.shape[1]
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)  # You also need to reshape y_val
    
    print("GML")

    time_start = timer()
    model = build_model(input_shape)
    
    # Add validation_data parameter in fit
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # predict and evaluate
    pred = model.predict(X_test[0])
    time_end = timer()
    total_time = time_end - time_start
    
    print(f"GML Time: {round(total_time, 4)}")
    print(pred, y_test[0])


if __name__ == "__main__":
    main()
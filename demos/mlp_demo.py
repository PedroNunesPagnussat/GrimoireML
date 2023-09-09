

# Your own library imports

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple
import numpy as np


from grimoireml.nn.models.sequential import Sequential
from grimoireml.nn.models.load_model import load_model
from grimoireml.nn.layers import Dense
from grimoireml.nn.functions.activation_functions import ReLU, Softmax
from grimoireml.nn.optimizers import SGD
from grimoireml.nn.functions.loss_functions import CCE
from grimoireml.nn.initializers.bias_initializers import ZerosBiasInitializer
from grimoireml.nn.initializers.weight_initializers import GlorotUniformWeightInitializer

# import grimoireml.functions.evaluation_functions as eval_functions


def load_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess the MNIST data set from scikit-learn."""
    
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    
    # Standardize the features
    X = X/255.
    
    # One-hot encode the target variable
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test



def build_model(input_shape: int) -> Sequential:
    """Build and return the model."""

    model = Sequential()
    model.add(Dense(input_shape, 1024, activation=ReLU()))
    model.add(Dense(1024, 512, activation=ReLU()))
    model.add(Dense(512, 256, activation=ReLU()))
    model.add(Dense(256, 128, activation=ReLU()))
    model.add(Dense(128, 64, activation=ReLU()))
    model.add(Dense(64, 32, activation=ReLU()))
    model.add(Dense(32, 10, activation=Softmax()))


    model.compile(optimizer=SGD(lr=0.05), loss=CCE())
    return model


def main():
    # set random seed
    np.random.seed(42)
    
    X_train, y_train, X_test, y_test = load_mnist_data()
    epochs = 20
    batch_size = 16


    input_shape = X_train.shape[1]


    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # predict and evaluate
    pred = model.predict(X_test[0])
    print(pred, y_test[0])
    print(np.argmax(pred), np.argmax(y_test[0]))
    print(sum(pred))
    # save model
    path_to_save = "super_cool_model.pkl"
    model.save_model(path_to_save)





if __name__ == "__main__":
    main()
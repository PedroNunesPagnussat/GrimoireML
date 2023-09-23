from icecream import ic
from grimoireml.NeuralNetwork.Layers import ReLU, Sigmoid, Dense
from grimoireml.NeuralNetwork.LossFunctions import MSELoss, BCELoss
from grimoireml.NeuralNetwork.Optimizers import SGD, Adam
from grimoireml.NeuralNetwork.Models import Sequential
from grimoireml.Functions.EvaluationFunctions import Accuracy
from dev_data_fetch import fetch_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

# Fetching data
X, y = fetch_data("breast_cancer")
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential(
    [
        Dense(16, input_shape=(X.shape[1],)),
        ReLU(),
        Dense(8),
        ReLU(),
        Dense(1),
        Sigmoid(),
    ]
)

model.build(loss=BCELoss(), optimizer=Adam())
model.fit(X, y, epochs=20, metrics=[Accuracy(classification_type="binary")], validation_data=(X_test, y_test), batch_size=1)
# ic(model.history.history)

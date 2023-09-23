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
import pandas as pd

#disable numpy warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)
# Fetching data
X, y = (
    pd.read_csv("data.csv").to_numpy()[:, :-1],
    pd.read_csv("data.csv").to_numpy()[:, -1],
)
X = StandardScaler().fit_transform(X)
# split in train, test and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Sequential(
    [
        Dense(16, input_shape=(X.shape[1],)),
        ReLU(),
        Dense(16),
        ReLU(),
        Dense(1),
        Sigmoid(),
    ]
)

model.build(loss=BCELoss(), optimizer=Adam(learning_rate=0.01))
model.fit(
    X_train,
    y_train,
    epochs=150,
    metrics=[Accuracy(classification_type="binary")],
    validation_data=(X_test, y_test),
    verbose=True,
    batch_size=len(X_train),
)
l, m = model.evaluate(X_test, y_test)


from tensorflow.keras.models import Sequential as KerasSequential
from tensorflow.keras.layers import Dense as KerasDense

# Disable Warning
import os
os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

keras_model = KerasSequential()
keras_model.add(KerasDense(16, activation="relu", input_shape=(X.shape[1],)))
keras_model.add(KerasDense(16, activation="relu"))
keras_model.add(KerasDense(1, activation="sigmoid"))
keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = keras_model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=len(X_train),
    validation_data=(X_test, y_test),
    verbose=0,
)



import skynet_ml
from skynet_ml.nn.models.sequential import Sequential as SkynetSequential
from skynet_ml.nn.layers import Dense as SkynetDense

print("SKYNET")
skynet_model = SkynetSequential()
skynet_model.add(SkynetDense(16, activation="relu", input_dim=X.shape[1]))
skynet_model.add(SkynetDense(16, activation="relu"))
skynet_model.add(SkynetDense(1, activation="sigmoid"))
skynet_model.compile(optimizer="adam", loss="bce", learning_rate=0.01)
skynet_model.fit(
    xtrain=X_train,
    ytrain=y_train,
    xval=X_test,
    yval=y_test,
    epochs=150,
    batch_size=len(X_train),
    metrics=["accuracy"]
)
print("skynet")
print(skynet_model.evaluate(X_test, y_test))
print("keras")
print(keras_model.evaluate(X_test, y_test))
print("grimoire")
print(l, m)

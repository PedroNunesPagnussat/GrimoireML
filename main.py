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

np.random.seed(42)
# Fetching data
X, y = pd.read_csv("data.csv").to_numpy()[:, :-1], pd.read_csv("data.csv").to_numpy()[:, -1]
X = StandardScaler().fit_transform(X)
# split in trainm, test and validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)


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

model.build(loss=BCELoss(), optimizer=Adam(learning_rate=0.001))
model.fit(X_train, y_train, epochs=150, metrics=[Accuracy(classification_type="binary")],
          validation_data=(X_test, y_test), verbose=1, batch_size=len(X_train))

# print(model.history.history["val_loss"][-1])
# r = model.predict(X_test)
# r = np.where(r > 0.5, 1, 0)
# r = r.reshape(-1)
# print(np.sum(r == y_test) / len(y_test))

#plot
import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.show()

plt.plot(model.history.history["Accuracy"])
plt.plot(model.history.history["val_Accuracy"])
plt.show()


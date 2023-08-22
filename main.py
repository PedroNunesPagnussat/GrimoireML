import numpy as np
from pedroflow.mlp import MLP
from pedroflow.layers import Input, Dense

# TODO - Implementar as classes abaixo (e as a cima tbm)
# from pedroflow.activations import Sigmoid, ReLU, Tanh, Softmax
# from pedroflow.losses import MSE, MAE, CrossEntropy
# from pedroflow.optimizers import SGD, Adam, RMSprop

np.random.seed(42)


DATA_PATH = "data/data.csv"
DATA = np.loadtxt(DATA_PATH, delimiter=",", dtype=float, skiprows=1)

X, y = DATA[:, :-1], DATA[:, -1]

INPUT_SHAPE = X.shape[1]
OUTPUT_SHAPE = 1

MSE = lambda: None
SGD = lambda: None

model = MLP()
model.add(Input(INPUT_SHAPE))
model.add(Dense(3, activation='relu'))
model.add(Dense(OUTPUT_SHAPE, activation='sigmoid'))
model.compile(loss="MSE", optimizer=SGD())

model.fit(X, y, epochs=1, batch_size=2)
"""




print(model.predict(X))







"""


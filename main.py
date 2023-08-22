import numpy as np
from pedroflow.neuralNetwork import Model
np.random.seed(42)


DATA_PATH = "data.csv"
DATA = np.loadtxt("data.csv",
                 delimiter=",", dtype=float, skiprows=1)

model = Model()


# disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from timeit import default_timer as timer
import numpy as np

from GrimoireML.grimoireml.nn.sequential import Sequential as GMLSequential
from GrimoireML.grimoireml.nn.layers import Dense
from GrimoireML.grimoireml.functions.activation_functions import Sigmoid, Linear
from GrimoireML.grimoireml.nn.optimizers import SGD
from GrimoireML.grimoireml.functions.loss_functions import MSE, BCE

from GRIMOIREML.grimoireml.nn.sequential import Sequential as GMLSequential
# set random seed
np.random.seed(42)


def steal_data_from_skl():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test = steal_data_from_skl()
epochs = 20
batch_size = 1
input_shape = X_train.shape[1]


print("GML")

time_start = timer()

model = GMLSequential()
model.add(Dense(input_shape, 16, activation=Sigmoid()))
model.add(Dense(16, 8, activation=Sigmoid(), weight_initializer="Glorot_uniform", bias_initializer="Zeros"))
model.add(Dense(8, 4,  activation=Sigmoid(), weight_initializer="Glorot_uniform", bias_initializer="Zeros"))
model.add(Dense(4, 2,  activation=Sigmoid(), weight_initializer="Glorot_uniform", bias_initializer="Zeros"))
model.add(Dense(2, 1,  activation=Linear(), weight_initializer="Glorot_uniform", bias_initializer="Zeros"))
model.compile(optimizer=SGD(lr=0.01), loss=MSE())
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# predict
pred = model.predict(X_test[0])
timr_end = timer()
total_time = timr_end - time_start
print(f"GML Time: {round(total_time, 4)}")
print(pred, y_test[0])


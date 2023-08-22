import numpy as np

def sigmoid(s_list):
    temp = np.array([])
    for s in s_list:
        temp = np.append(temp, 1 / (1 + np.exp(-s)))
    return temp

def sigmoid_derivative(x):
    return x * (1 - x)

def cost_function(y_hat, y):
    return  0.5 * np.sum(np.square(y_hat - y))

def cost_function_derivative(y_hat, y):
    return y_hat - y 


DATA_PATH = "data.csv"
DATA = np.loadtxt("data.csv",
                 delimiter=",", dtype=float, skiprows=1)

NEURON_NUMBERS = np.array([2, 3, 1])

WEIGTHS = np.load('weights.npy', allow_pickle=True)

EPOCHS = 1
LR = 0.01

def foward_pass():
    pass

def backward_pass():
    pass

def train(EPOCHS=1, LR=0.01, BATCH_SIZE=1):
    for epoch in EPOCHS:
        pass
            


for epoch in range(EPOCHS):
    LOSS_LIST = np.array([])
    for data in DATA:

        HIDDEN_ACTIVATIONS = []
        DELTAS = []
        GRADS = []
        X, y = data[:-1].T, data[-1]

        # FORWARD PASS
        for i, layer in enumerate(WEIGTHS):
            if i == 0:
                o = sigmoid(np.dot(layer, X))
            else:
                o = sigmoid(np.dot(layer, HIDDEN_ACTIVATIONS[-1]))

            # Append the activations of the current layer to the list
            HIDDEN_ACTIVATIONS.append(o)

        # Loss
        loss = cost_function(HIDDEN_ACTIVATIONS[-1], y)
        LOSS_LIST = np.append(LOSS_LIST, loss)

        # BACKWARD PASS
        for i, layer in reversed(list(enumerate(WEIGTHS))):
            if i == len(WEIGTHS) - 1:
                delta = cost_function_derivative(HIDDEN_ACTIVATIONS[i], y) * sigmoid_derivative(HIDDEN_ACTIVATIONS[i])
            else:
                delta = np.dot(WEIGTHS[i + 1].T, DELTAS[-1]) * sigmoid_derivative(HIDDEN_ACTIVATIONS[i])


            DELTAS.append(delta)
        DELTAS = np.flip(DELTAS)


        for i in range(len(WEIGTHS)):

            if i == 0:
                grad = np.dot(DELTAS[0].reshape(-1, 1), X.reshape(1, -1)) 
            else:
                grad = np.dot(DELTAS[i].reshape(-1, 1), HIDDEN_ACTIVATIONS[i - 1].reshape(1, -1))
            GRADS.append(grad)


        # Adjust weights

        for i in range(len(WEIGTHS)):
            WEIGTHS[i] -= LR * GRADS[i]
            
print(WEIGTHS)
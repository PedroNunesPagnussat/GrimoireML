# Multi-Layer Perceptron (MLP) from Scratch

This project implements a simple Multi-Layer Perceptron (MLP) from scratch using NumPy. It includes functions for the sigmoid activation, its derivative, a mean squared error cost function, and its derivative. The code covers forward and backward passes through the neural network and updates the weights using calculated gradients.

## Overview

The code consists of the following main components:

- **Activation Functions**: Sigmoid activation function along with its derivative.
- **Cost Function**: Mean squared error cost function and its derivative.
- **Forward Pass**: Calculation of activations for the hidden layers using the sigmoid activation function.
- **Backward Pass**: Calculation of gradients using the chain rule, activation, and cost function derivatives.
- **Weight Update**: Update of the weights using the calculated gradients and a learning rate.

## Usage

1. **Data Initialization**: Define the input data and target values as NumPy arrays.
2. **Network Configuration**: Set the neuron numbers for each layer and initial weights.
3. **Training Parameters**: Define the number of epochs and learning rate for training.
4. **Run Training Loop**: The training loop will run through the specified number of epochs, performing forward and backward passes, calculating the loss, and updating the weights.

## Code Structure

- `sigmoid(s)`: Sigmoid activation function.
- `sigmoid_derivative(x)`: Derivative of the sigmoid activation function.
- `cost_function(y_hat, y)`: Mean squared error cost function.
- `cost_function_derivative(y_hat, y)`: Derivative of the mean squared error cost function.
- `DATA`: Input data and target values.
- `NEURON_NUMBERS`: Number of neurons in each layer.
- `WEIGTHS`: Initial weights for the network.
- `EPOCHS`: Number of epochs for training.
- `LR`: Learning rate.

## Requirements

- Python 3.x
- NumPy

## License

This project is open-source and available under the MIT License.

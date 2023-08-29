import numpy as np
from grimoireml.functions import loss_functions

class Sequential():

    def __init__(self) -> None:
        self.loss = None
        self.loss_derivative = None
        self.optimizer = None
        self.layers = np.array([], dtype=object)


    def add(self, layer):
        self.layers = np.append(self.layers, layer)
    
    def compile(self, loss, optimizer):
        self.loss, self.loss_derivative = loss_functions.get_loss_function(loss)
        self.optimizer = optimizer

        input_shape = self.layers[0].input_shape
        for i, layer in enumerate(self.layers[1:]):
            layer.initialize_weights(input_shape)
            layer.initialize_biases()
            input_shape = layer.neurons


    def fit(self, X, y, epochs=1, lr=0.01, batch_size=1):
        for epoch in range(epochs):
           # print(f"Epoch {epoch + 1}/{epochs}")
            
            for i in range(0, len(X), batch_size):

               # print(f"Batch {i + 1}/{len(X) // batch_size}")

                self.layers[0].output = X[i:i+batch_size]

                batch_y = y[i:i+batch_size]


                y_pred = self.__forward(self.layers[0].output)
                self.__backward(batch_y, y_pred)
                self.__update(lr)


    
    def __forward(self, X):
        inputs = X
        for layer in self.layers[1:]:
            z = np.dot(inputs, layer.weights) + layer.biases
            z = np.array(z, dtype=np.float)
            layer.output = np.apply_along_axis(layer.activation, 1, z)  # Apply activation along the correct axis
            # layer.output_derivative = np.apply_along_axis(layer.activation_derivative, 1, layer.output)
            inputs = layer.output

        return inputs

    
    def __backward(self, y_true, y_pred):
        y_true = y_true.reshape(y_pred.shape)
        error = self.loss_derivative(y_true, y_pred)

        # Delta for the last (output) layer
        output_layer = self.layers[-1]
        output_layer.delta = error * np.apply_along_axis(output_layer.activation_derivative, 1, output_layer.output)

        # Iterate through hidden layers in reverse order
        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Compute delta for the current hidden layer
            error = np.dot(next_layer.delta, next_layer.weights.T)
            layer.delta = error * np.apply_along_axis(layer.activation_derivative, 1, layer.output)


    def __compute_gradients(self):
        gradients = []

        for i, layer in enumerate(self.layers[1:]):
            prev_layer = self.layers[i]
            weights_gradients = np.dot(prev_layer.output.T, layer.delta)
            biases_gradients = np.sum(layer.delta, axis=0)


           
            # Append gradients as a tuple
            gradients.append((weights_gradients, biases_gradients))

        # Return the gradients in the correct order (from input to output layers)
        return gradients
    
    def __update(self, lr):
        gradients = self.__compute_gradients()
        for i, (weight_gradients, bias_gradients) in enumerate(gradients, start=1):
            layer = self.layers[i]

            # Update weights and biases
            layer.weights -= lr * weight_gradients
            layer.biases -= lr * bias_gradients

    def predict(self, X):
        inputs = X
        for layer in self.layers[1:]:  # Skip the input layer
            z = np.dot(inputs, layer.weights) + layer.biases
            z = np.array(z, dtype=np.float)
            inputs = np.apply_along_axis(layer.activation, 1, z)  # Apply activation function
        return inputs

        
    def __str__(self) -> str:
        s = f"MLP with {len(self.layers)} layers:\n"
        s += f"Loss: {self.loss}\n"
        s += f"Optimizer: {self.optimizer}\n"
        for i, layer in enumerate(self.layers):
            s += f"Layer {i}: {layer}\n"

        return s
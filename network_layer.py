import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        return self.activation.forward(self.z)

    def backward(self, gradient, learning_rate):
        grad = self.activation.backward(gradient)
        d_weights = self.input.T @ grad
        d_biases = np.sum(grad, axis=0, keepdims=True)
        d_input = grad @ self.weights.T
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input

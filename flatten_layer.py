import numpy as np

class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, gradient, learning_rate=None):
        return gradient.reshape(self.input_shape)

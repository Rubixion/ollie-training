import numpy as np

class Relu:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, gradient):
        return gradient * (self.input > 0)


class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, gradient):
        return gradient * self.output * (1 - self.output)

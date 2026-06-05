import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x, y, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(x)
            loss = np.mean((output - y) ** 2)
            gradient = 2 * (output - y) / y.size
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

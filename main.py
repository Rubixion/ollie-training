from network import Network
from activation_function import Sigmoid, Relu
from network_layer import Layer
import numpy as np

def main():
    np.random.seed(42)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    network = Network()
    network.add(Layer(2, 8, Relu()))
    network.add(Layer(8, 1, Sigmoid()))

    network.train(x, y, learning_rate=0.1, epochs=20000)

    print(np.round(network.forward(x), 3))

if __name__ == "__main__":    
    main()
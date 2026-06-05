import numpy as np
import matplotlib.pyplot as plt
from network import Network
from network_layer import Layer
from conv_layer import ConvLayer
from pool_layer import MaxPool
from flatten_layer import Flatten
from activation_function import Relu, Sigmoid

def make_data(n_each=50):
    images, labels = [], []
    for _ in range(n_each):
        img = np.zeros((8, 8))
        img[:4, :] = 1.0        # horizontal stripe = label 0
        images.append(img)
        labels.append(0)
    for _ in range(n_each):
        img = np.zeros((8, 8))
        img[:, :4] = 1.0        # vertical stripe = label 1
        images.append(img)
        labels.append(1)
    x = np.array(images)                    # shape (100, 8, 8)
    y = np.array(labels).reshape(-1, 1)     # shape (100, 1)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

def main():
    np.random.seed(42)
    x, y = make_data()

    network = Network()
    network.add(ConvLayer(8, 3, Relu()))    # (100,8,8) -> (100,6,6,8)
    network.add(MaxPool(2, 2))              # -> (100,3,3,8)
    network.add(Flatten())                  # -> (100,72)
    network.add(Layer(72, 16, Relu()))      # -> (100,16)
    network.add(Layer(16, 1, Sigmoid()))    # -> (100,1)

    # show 8 sample images before training
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle("Sample training images", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title("horizontal" if y[i][0] == 0 else "vertical")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    network.train(x, y, learning_rate=0.01, epochs=2000)

    predictions = np.round(network.forward(x))
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")

    # show predictions on first 8 images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle("Predictions after training", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i], cmap="gray", vmin=0, vmax=1)
        pred = "vertical" if predictions[i][0] == 1 else "horizontal"
        true = "vertical" if y[i][0] == 1 else "horizontal"
        color = "green" if pred == true else "red"
        ax.set_title(f"pred: {pred}", color=color)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

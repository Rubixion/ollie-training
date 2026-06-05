import numpy as np
from network import Network
from network_layer import Layer
from conv_layer import ConvLayer
from pool_layer import MaxPool
from flatten_layer import Flatten
from activation_function import Relu, Sigmoid

IMAGE_SIZE = 32  # images are resized to 32x32 before being fed in


class SiameseNetwork:
    """
    Two images are passed through the same backbone (shared weights).
    The absolute difference between their embeddings is fed into a small
    head that outputs a similarity score: 1 = same person, 0 = different.

    Weight sharing is achieved by concatenating both images into one batch,
    running one forward pass, then splitting the embeddings and computing
    the difference.  The same trick works for backpropagation.
    """

    def __init__(self, embedding_size=32):
        # Shared CNN backbone
        # Input: (N, 32, 32, 1)
        # After ConvLayer(8, 3):  (N, 30, 30, 8)
        # After MaxPool(2, 2):    (N, 15, 15, 8)
        # After ConvLayer(16, 3): (N, 13, 13, 16)
        # After MaxPool(2, 2):    (N,  6,  6, 16)  → flat: 576
        self.backbone = Network()
        self.backbone.add(ConvLayer(8, 3, Relu()))
        self.backbone.add(MaxPool(2, 2))
        self.backbone.add(ConvLayer(16, 3, Relu()))
        self.backbone.add(MaxPool(2, 2))
        self.backbone.add(Flatten())
        self.backbone.add(Layer(576, 128, Relu()))
        self.backbone.add(Layer(128, embedding_size, Relu()))

        # Head: maps |emb1 - emb2| -> similarity score in (0, 1)
        self.head = Layer(embedding_size, 1, Sigmoid())

    def forward(self, x1, x2):
        N = len(x1)
        embeddings = self.backbone.forward(np.concatenate([x1, x2]))
        emb1, emb2 = embeddings[:N], embeddings[N:]
        pred = self.head.forward(np.abs(emb1 - emb2))
        return pred, emb1, emb2

    def train(self, x1, x2, y, learning_rate=0.001, epochs=500):
        N = len(y)
        for epoch in range(epochs):
            # ---- forward ----
            embeddings = self.backbone.forward(np.concatenate([x1, x2]))
            emb1, emb2 = embeddings[:N], embeddings[N:]
            diff = np.abs(emb1 - emb2)
            pred = self.head.forward(diff)

            # binary cross-entropy
            eps = 1e-8
            loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

            # ---- backward ----
            d_pred = -(y / (pred + eps) - (1 - y) / (1 - pred + eps)) / N
            d_diff = self.head.backward(d_pred, learning_rate)

            # gradient through |emb1 - emb2|
            sign = np.sign(emb1 - emb2)
            d_emb = np.concatenate([d_diff * sign, d_diff * -sign])

            # backprop through shared backbone (single pass, correct weight-sharing)
            grad = d_emb
            for layer in reversed(self.backbone.layers):
                grad = layer.backward(grad, learning_rate)

            if epoch % 50 == 0:
                acc = np.mean((pred >= 0.5) == y)
                print(f"Epoch {epoch:4d}  Loss: {loss:.4f}  Accuracy: {acc * 100:.0f}%")

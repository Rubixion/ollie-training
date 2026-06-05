import numpy as np

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        N, H, W, C = x.shape
        P = self.pool_size
        S = self.stride
        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        out = np.zeros((N, H_out, W_out, C))
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*S:i*S+P, j*S:j*S+P, :]
                out[:, i, j, :] = np.max(patch, axis=(1, 2))
        return out

    def backward(self, gradient, learning_rate=None):
        N, H, W, C = self.input.shape
        P = self.pool_size
        S = self.stride
        _, H_out, W_out, _ = gradient.shape

        d_input = np.zeros_like(self.input)
        for i in range(H_out):
            for j in range(W_out):
                patch = self.input[:, i*S:i*S+P, j*S:j*S+P, :]
                max_vals = np.max(patch, axis=(1, 2), keepdims=True)
                mask = (patch == max_vals)
                d_input[:, i*S:i*S+P, j*S:j*S+P, :] += mask * gradient[:, i:i+1, j:j+1, :]
        return d_input

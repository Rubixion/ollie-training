import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, activation, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.biases = None

    def _initialize(self, channels):
        fan_in = self.filter_size * self.filter_size * channels
        self.filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size, channels) * np.sqrt(2 / fan_in)
        self.biases = np.zeros(self.num_filters)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, :, :, np.newaxis]
        if self.filters is None:
            self._initialize(x.shape[3])

        self.input = x
        N, H, W, C = x.shape
        F = self.filter_size
        S = self.stride
        P = self.padding

        if P > 0:
            x = np.pad(x, ((0, 0), (P, P), (P, P), (0, 0)))
        self.x_padded = x

        H_out = (H + 2 * P - F) // S + 1
        W_out = (W + 2 * P - F) // S + 1
        z = np.zeros((N, H_out, W_out, self.num_filters))

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*S:i*S+F, j*S:j*S+F, :]
                z[:, i, j, :] = np.tensordot(patch, self.filters, axes=([1, 2, 3], [1, 2, 3])) + self.biases

        self.z = z
        return self.activation.forward(z)

    def backward(self, gradient, learning_rate):
        grad = self.activation.backward(gradient)
        N, H_out, W_out, _ = grad.shape
        F = self.filter_size
        S = self.stride

        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(grad, axis=(0, 1, 2))
        d_x_padded = np.zeros_like(self.x_padded)

        for i in range(H_out):
            for j in range(W_out):
                patch = self.x_padded[:, i*S:i*S+F, j*S:j*S+F, :]
                g = grad[:, i, j, :]
                d_filters += np.tensordot(g, patch, axes=([0], [0]))
                d_x_padded[:, i*S:i*S+F, j*S:j*S+F, :] += np.tensordot(g, self.filters, axes=([1], [0]))

        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        P = self.padding
        if P > 0:
            return d_x_padded[:, P:-P, P:-P, :]
        return d_x_padded

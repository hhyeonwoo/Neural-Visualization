import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNet:
    def __init__(self, layer_sizes, lr=0.1):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.num_layers = len(layer_sizes)
        self.weights = {}
        self.biases = {}
        self.activations = {}
        self.zs = {}

        for i in range(1, self.num_layers):
            self.weights[i] = 0.1 * np.random.randn(layer_sizes[i], layer_sizes[i - 1])
            self.biases[i] = np.zeros((layer_sizes[i],))

    def sigmoid(self, x, derivative=False):
        f = 1 / (1 + np.exp(-x))
        return f * (1 - f) if derivative else f

    def forward(self, x):
        self.activations[0] = x
        for i in range(1, self.num_layers):
            self.zs[i] = self.weights[i] @ self.activations[i - 1] + self.biases[i]
            self.activations[i] = self.sigmoid(self.zs[i])
        return self.activations[self.num_layers - 1]

    def backward(self, y):
        grads = {}
        delta = (self.activations[self.num_layers - 1] - y) * self.sigmoid(self.zs[self.num_layers - 1], derivative=True)
        grads[self.num_layers - 1] = delta

        for i in reversed(range(1, self.num_layers - 1)):
            delta = self.weights[i + 1].T @ delta * self.sigmoid(self.zs[i], derivative=True)
            grads[i] = delta

        return grads

    def update_weights(self, grads):
        for i in range(1, self.num_layers):
            grad_w = np.outer(grads[i], self.activations[i - 1])
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grads[i]

    def train_stream(self, X, Y, ax):
        for index, (x, y) in enumerate(zip(X, Y)):
            self.forward(x)
            grads = self.backward(y)
            self.update_weights(grads)
            self.visualize_weights(ax)
            print(f"Rows left: {len(X) - index}")

    def visualize_weights(self, ax_list):
        for i, ax in enumerate(ax_list, start=1):
            ax.clear()
            ax.set_title(f"Weights Layer {i}")
            ax.imshow(self.weights[i], cmap='viridis', aspect='auto')
        plt.pause(0.001)



np.random.seed(0)
inputs = np.random.rand(200, 9)
targets = np.random.rand(200, 3)


layer_sizes = [9, 7, 5, 3]
model = SimpleNeuralNet(layer_sizes=layer_sizes, lr=0.1)


num_layers = len(layer_sizes) - 1
fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
if num_layers == 1:
    axes = [axes]


model.train_stream(inputs, targets, ax=axes)
plt.show()

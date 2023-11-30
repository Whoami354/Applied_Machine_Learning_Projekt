import numpy as np


class NeuronalNetwork:
    def __init__(self, shapes):
        self.layers = []
        self.weights = []
        self.biases = []
        self.a = self.activation

        for idx in range(len(shapes) - 1):
            self.layers.append(np.zeros((shapes[idx], 1)))
            self.weights.append(np.random.rand(shapes[(idx + 1)], shapes[idx]))
            self.biases.append(np.random.rand(shapes[idx], 1))

    def feet_forward(self, input_vector):
        self.layers[0] = input_vector
        for idx in range(1, len(self.layers)):
            self.layers[idx] = self.weights[idx - 1] @ self.layers[idx - 1]
            self.layers[idx] = self.a(self.layers[idx])
            self.layers[idx].shape = (self.layers[idx].shape[0], 1)

        return self.a(self.weights[-1] @ self.layers[-1])

    def activation(self, vector_x):
        return 1 / (1 + np.exp(-vector_x))

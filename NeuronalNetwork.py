import numpy as np

class NeuronalNetwork:
    # shapes: A list of integers representing the number of neurons in each layer of the neural network.
    def __init__(self, shapes):
        # Initialize the lists for the layers (as vectors) and weights (as matrices)
        self.layers = []
        self.weights = []

        # Defines the activation function
        self.a = self.activation

        # The structure of the network (number of neurons per layer)
        self.shapes = shapes

        # Initialize the layers and weights
        for idx in range(len(shapes) - 1):
            self.layers.append(np.zeros((shapes[idx], 1)))  # Initializes each layer with zeros
            self.weights.append(np.random.rand(shapes[(idx + 1)], shapes[idx]))  # Initializes the weights with random values

    def feed_forward(self, input_vector):
        # Sets the input vector as the first layer
        self.layers[0] = input_vector
        # Calculates the output vector by forward propagation
        for idx in range(1, len(self.layers) - 1):
            # Multiplies the previous layer by the weights and applies the activation function
            self.layers[idx] = self.a(self.weights[idx - 1] @ self.layers[idx - 1])
            self.layers[idx].shape = (self.layers[idx].shape[0], 1)  # Ensures that the shape is correct

        # Returns the calculated output vector, which should not have n number of columns
        return self.a(self.weights[-1] @ self.layers[-1])

    def activation(self, vector_x):
        # Calculates the sigmoid function for the vector
        return 1 / (1 + np.exp(-vector_x))

    def copy(self):
        cloned_Brain = NeuronalNetwork(self.shapes)
        cloned_Brain.layers = self.layers.copy()
        cloned_Brain.weights = self.weights.copy()
        cloned_Brain.a = self.activation
        cloned_Brain.shapes = self.shapes.copy()

        # Returns the new instance
        return cloned_Brain

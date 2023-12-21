import numpy as np

# Neuronales netz ist Simulation von Biologischem Gehirn
class NeuronalNetwork:
    def __init__(self, shapes):
        # schichten vom neuronalen netzwerk
        self.layers = []
        # die Verbindungen (synapsen) zwischen den layers
        self.weights = []
        # Vorstellung der aktivierung von den Neuronen
        self.a = self.activation
        # der Aufbau des Gehirns
        self.shapes = shapes

        for idx in range(len(shapes) - 1):
            # Wir initialisieren die Schichten und die synapsen
            # layers machen wir mit 0 voll
            self.layers.append(np.zeros((shapes[idx], 1)))
            # weights machen wir mit zufälligen werten rein, weil wir diversität haben wollen.
            # Diversität, damit wir viele Ansätze haben das Problem zu lösen
            self.weights.append(np.random.rand(shapes[(idx + 1)], shapes[idx]))

    def feed_forward(self, input_vector):
        """Weiterleitung zwischen den Synapsen durch neuronales Netzwerk"""
        # Wir multiplizieren Vectoren mit Matrizen
        # Kopiert unseren input vector in unseren Layer rein.
        self.layers[0] = input_vector
        for idx in range(1, len(self.layers)):
            # wir gehen durch die einzelnen schichten durch und multiplizieren die
            #vorangegagen schichten mit den vorangegangenen weights
            # weights = Matrix layers = Vector
            # @ ist von numpy wie man matrizen multipliziert.
            self.layers[idx] = self.weights[idx - 1] @ self.layers[idx - 1]
            #lassen es durch die aktivierungsfunktion laufen
            self.layers[idx] = self.a(self.layers[idx])
            self.layers[idx].shape = (self.layers[idx].shape[0], 1)
        # wir rechnen das Ergebnis für den letzten Layer aus und geben das aus.
        return self.a(self.weights[-1] @ self.layers[-1])

    def activation(self, vector_x):
        # sigmoid Rechenformel
        return 1 / (1 + np.exp(-vector_x))

    def copy(self):
        # wir machen eine deepcopy von dem Gehirn
        cloned_Brain = NeuronalNetwork(self.shapes)
        cloned_Brain.layers = self.layers.copy()
        cloned_Brain.weights = self.weights.copy()
        cloned_Brain.a = self.activation
        cloned_Brain.shapes = self.shapes.copy()
        return cloned_Brain

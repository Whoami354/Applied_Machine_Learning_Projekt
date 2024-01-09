import numpy as np

class NeuronalNetwork:
    #shapes: Eine Liste von Ganzzahlen, die die Anzahl der Neuronen in jeder Schicht des neuronalen Netzwerks repräsentieren.
    def __init__(self, shapes):
        # Initialisiere die Listen für die Schichten (layers, als Vektor) und Gewichte (weights, also matrizen)
        self.layers = []
        self.weights = []

        # Definiert die Aktivierungsfunktion
        self.a = self.activation

        # Die Struktur des Netzwerks (Anzahl der Neuronen pro Schicht)
        self.shapes = shapes

        # Initialisiere die Schichten und Gewichte
        for idx in range(len(shapes) - 1):
            self.layers.append(np.zeros((shapes[idx], 1)))  # Initialisiert jede Schicht mit Nullen
            self.weights.append(
                np.random.rand(shapes[(idx + 1)], shapes[idx]))  # Initialisiert die Gewichte mit Zufallswerten

    def feed_forward(self, input_vector):
        # Setzt den Eingabevektor als erste Schicht
        self.layers[0] = input_vector
        # Berechnet den Ausgabevektor durch Vorwärtspropagierung
        for idx in range(1, len(self.layers) - 1):
            # Multipliziert die vorherige Schicht mit den Gewichten und wendet die Aktivierungsfunktion an
            self.layers[idx] = self.a(self.weights[idx - 1] @ self.layers[idx - 1])
            self.layers[idx].shape = (self.layers[idx].shape[0], 1)  # Stellt sicher, dass die Form korrekt ist

        # Gibt den berechneten Ausgabevektor zurück
        return self.a(self.weights[-1] @ self.layers[-1])

    def activation(self, vector_x):
        # Berechnet die Sigmoid-Funktion für den Vektor
        return 1 / (1 + np.exp(-vector_x))

    def copy(self):
        cloned_Brain = NeuronalNetwork(self.shapes)
        cloned_Brain.layers = self.layers.copy()
        cloned_Brain.weights = self.weights.copy()
        cloned_Brain.a = self.activation
        cloned_Brain.shapes = self.shapes.copy()

        # Gibt die neue Instanz zurück
        return cloned_Brain

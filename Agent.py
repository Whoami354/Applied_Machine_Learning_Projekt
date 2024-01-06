from NeuronalNetwork import NeuronalNetwork as nn  # Importiert die NeuronalNetwork-Klasse

# Definition der Klasse Agent
class Agent:
    def __init__(self):
        # Initialisierungsmethode für einen neuen Agenten
        self.brain = nn([8, 4])  # Erstellt ein neuronales Netzwerk mit 8 Eingangsneuronen und 4 Ausgangsneuronen
        self.reward = 0  # Initialisiert die Belohnung des Agenten mit 0

    def fuehre_aktion(self, input_vector):
        # Methode, um das neuronale Netzwerk des Agenten eine Aktion ausführen zu lassen basierend auf einem Eingabevektor
        return self.brain.feed_forward(input_vector)  # Führt die Vorwärtspropagierung im neuronalen Netzwerk aus

    def __str__(self):
        return str(self.reward)

    def clone(self):
        # Methode, um eine Kopie dieses Agenten zu erstellen
        child = Agent()  # Erstellt einen neuen Agenten
        child.brain = self.brain.copy()  # Kopiert das "Gehirn" (neuronales Netzwerk) des aktuellen Agenten
        return child  # Gibt den neuen Agenten zurück

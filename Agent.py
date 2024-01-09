from NeuronalNetwork import NeuronalNetwork as nn

# Definition der Klasse Agent
class Agent:
    def __init__(self):
        # Das neuronale Netzwerk hat 8 Eingabeneuronen und 4 Ausgabeneuronen
        """
        8 Eingabeneuronen:
        X-distanz vom Ziel
        Y-distanz vom Ziel
        X-Geschwindigkeit
        Y-Geschwindigkeit
        Neigung des schiffes
        Winkelgeschwindigkeit des Schiffes
        Linkes Bein auf den Boden (Wahr/Falsch)
        Rechtes Bein auf den Boden (Wahr/Falsch)
        """
        self.brain = nn([8, 4])
        # Belohnung wird als Wert für die Leistung verwendet
        self.reward = 0

    def execute_action(self, input_vector):
        # Methode, um eine Aktion basierend auf einem gegebenen Eingabevektor durchzuführen
        # Die Methode verwendet die feed_forward-Funktion des neuronalen Netzwerks, um die Aktion zu bestimmen
        return self.brain.feed_forward(input_vector)

    def __str__(self):
        return str(self.reward)

    def clone(self):
        # Methode, um eine Kopie (Klon) des Agenten zu erstellen
        child = Agent()  # Erstellt einen neuen Agenten
        child.brain = self.brain.copy()  # Kopiert das Gehirn (neuronale Netzwerk) des aktuellen Agenten in den neuen Agenten
        return child  # Gibt den neuen Agenten zurück

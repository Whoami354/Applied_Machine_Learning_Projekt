import pickle
import random

from Agent import Agent as ag
import numpy as np

# Definition der Klasse GA (Genetischer Algorithmus)
class GA:
    def __init__(self):
        # Initialisierung der Eigenschaften des genetischen Algorithmus
        self.Goal = 150  # Abbruchbedingung, Zielbelohnung
        self.mutation_rate = 0.10  # Mutationswahrscheinlichkeit
        self.population = []  # Liste der Agenten (Bevölkerung)
        self.average_reward = []  # Liste der durchschnittlichen Belohnungen
        self.generation_number = 0  # Aktuelle Generation
        self.stop = False  # Abbruchbedingung (Flag)
        self.population_size = 1200  # Größe der Bevölkerung

        # Initialisierung der Anfangsbevölkerung
        for idx in range(self.population_size):
            self.population.append(ag())  # Fügt neue Agenten zur Bevölkerung hinzu

    #def reproduce(self):
    #    # Methode zur Reproduktion der Bevölkerung ohne Crossover
    #    print(self.generation_number)
    #    # Sortiert die Agenten nach ihrer Belohnung in absteigender Reihenfolge
    #    sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
    #    # Berechnet die Gesamtbelohnung und den Durchschnitt
    #    rewards = sum(map(lambda agent: agent.reward, sorted_agents))
    #    print("rewards:", rewards / len(self.population))
    #    # Bestimmt die Anzahl der Agenten, die zu den besten 20% gehören
    #    n = int(len(sorted_agents) * 0.2)
    #    top_20 = sorted_agents[:n]  # Selektiert die besten 20%
    #    new_population = []
    #    for idx in range(self.population_size):
    #        candidate = top_20[idx % n]
    #        child = self.mutation(candidate)  # Wendet Mutation auf den Kandidaten an
    #        new_population.append(child)  # Fügt das Kind zur neuen Bevölkerung hinzu
    #    self.generation_number += 1  # Erhöht die Generationenzahl
    #    self.population = new_population  # Aktualisiert die Bevölkerung

    def mutation(self, a):
        # Methode zur Durchführung einer Mutation auf einem Agenten
        agent = a.clone()  # Erstellt eine Kopie des Agenten
        for idx in range(len(agent.brain.weights)):
            matrix = agent.brain.weights[idx]  # Zugriff auf die Gewichtsmatrix
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rand = np.random.uniform()  # Erzeugt eine Zufallszahl
                    # Wendet Mutation an, wenn die Zufallszahl kleiner als die Mutationsrate ist
                    if rand < self.mutation_rate:
                        matrix[i, j] += np.random.uniform(-0.015, 0.015)  # Verändert das Gewicht leicht
        return agent  # Gibt den mutierten Agenten zurück

    def reproduce_crossover(self):
        # Methode zur Reproduktion der Bevölkerung mit Crossover
        print(self.generation_number)
        sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
        self.average_rewards(self.Goal)  # Aktualisiert die Belohnungen und prüft die Abbruchbedingung
        n = int(len(sorted_agents) * 0.1)  # Bestimmt die Anzahl der Top-10%
        top_10 = sorted_agents[:n]  # Selektiert die besten 10%
        new_population = []
        for idx in range(self.population_size):
            random_first_parent = top_10[random.randrange(n)] # Wählt zufällig den ersten Elternteil
            random_second_parent = top_10[random.randrange(n)] # Wählt zufällig den zweiten Elternteil
            child = self.crossover(random_first_parent, random_second_parent)  # Erzeugt ein Kind durch Crossover
            child = self.mutation(child)  # Wendet Mutation auf das Kind an
            new_population.append(child)  # Fügt das Kind zur neuen Bevölkerung hinzu
        self.population = new_population  # Aktualisiert die Bevölkerung
        self.generation_number += 1  # Erhöht die Generationenzahl

    def average_rewards(self, threshold):
        # Methode zur Berechnung der durchschnittlichen Belohnung und Überprüfung der Abbruchbedingung
        rewards = sum(map(lambda agent: agent.reward, self.population))
        average_reward = rewards / len(self.population)
        self.average_reward.append(average_reward)
        if average_reward > threshold:  # Überprüft, ob die durchschnittliche Belohnung das Ziel überschritten hat
            self.stop = True  # Setzt das Abbruchflag
            self.force_rewards()  # Schreibt die Belohnungen in eine Datei
        print("rewards:", self.average_reward[-1])

    def force_rewards(self):
        # Methode zum Schreiben der durchschnittlichen Belohnungen in eine Datei
        with open("rewards.txt", "w") as file:
            for line in self.average_reward:
                file.write(str(line) + "\n")  # Schreibt jede Belohnung in die Datei

    def crossover(self, first_partent, second_parent):
        # Methode zur Durchführung eines Crossovers zwischen zwei Agenten
        child = ag()  # Erstellt einen neuen Agenten (Kind)
        parents = [first_partent, second_parent]  # Definiert die Elternteile
        for idx in range(len(child.brain.weights)):
            matrix = child.brain.weights[idx]  # Zugriff auf die Gewichtsmatrix des Kindes
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    parent = parents[random.randrange(2)]  # Wählt zufällig einen Elternteil
                    parent_value = parent.brain.weights[idx][i, j]  # Holt das entsprechende Gewicht
                    matrix[i, j] = parent_value  # Setzt das Gewicht des Elternteils
        return child  # Gibt das Kind zurück

    def checkpoint(self):
        # Methode zum Speichern der aktuellen Bevölkerung in einer Datei
        with open("checkpoint.pickle","wb") as file:
            pickle.dump(self.population, file)  # Schreibt die Bevölkerung in die Datei

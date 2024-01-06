import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Laden und Plotten der Belohnungen aus einer Datei
with open("rewards.txt", "r") as reward_file:
    rewards = reward_file.readlines()
    y_values = []
    for reward in rewards:
        # Konvertiert jede Belohnung in eine Fließkommazahl, rundet auf drei Dezimalstellen ab
        y_values.append(np.round(float(reward), decimals=3))
    x_values = range(len(rewards))  # Erstellt eine Liste von x-Werten entsprechend der Anzahl der Belohnungen
    plt.plot(x_values, y_values, "-")  # Plotte die Belohnungen als Linie
    plt.show()

env = gym.make("LunarLander-v2", render_mode='human')

# Laden der Population aus einer Pickle-Datei
with open("checkpoint.pickle", "rb") as file:
    population = pickle.load(file)  # Lädt die Population
    population_size = len(population)  # Bestimmt die Größe der Population

# Startet die Haupt-Simulationschleife
while True:
    for idx in range(population_size):
        agent = population[idx]  # Wählt den aktuellen Agenten
        observation, info = env.reset()  # Setzt die Umgebung zurück und erhält die erste Beobachtung
        for _ in range(1000):  # Simuliert bis zu 1000 Schritte
            action = np.argmax(agent.fuehre_aktion(observation))  # Bestimmt die Aktion basierend auf der Agentenrichtlinie
            observation, reward, terminated, truncated, info = env.step(action)  # Führt die Aktion aus und erhält neue Zustände
            agent.reward = reward  # Aktualisiert die Belohnung des Agenten
            # Beendet die Episode, wenn das Spiel vorbei ist
            if terminated or truncated:
                break

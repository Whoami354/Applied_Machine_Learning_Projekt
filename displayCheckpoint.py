import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Laden und Plotten der Belohnungen aus einer Datei
with open("rewards.txt", "r") as reward_file:
    rewards = reward_file.readlines()  # Lese alle Zeilen aus der Datei
    y_values = [np.round(float(reward), decimals=3) for reward in rewards]  # Konvertiere die Belohnungen in Float und runde auf drei Dezimalstellen ab
    x_values = range(len(rewards))  # Erstelle eine Liste von x-Werten entsprechend der Anzahl der Belohnungen
    plt.plot(x_values, y_values, "-")  # Plotte die Belohnungen als Linie
    plt.show()  # Zeige den Plot an

# Erstellen einer "LunarLander-v2" Umgebung mit menschlicher Darstellung
env = gym.make("LunarLander-v2", render_mode='human')

# Laden der Population aus einer Pickle-Datei
with open("checkpoint.pickle", "rb") as file:
    population = pickle.load(file)  # Lade die Population
    population_size = len(population)  # Bestimme die Größe der Population

# Starte die Haupt-Simulationschleife
while True:
    # Iteriere über jeden Agenten in der Population
    for idx in range(population_size):
        agent = population[idx]  # Wähle den aktuellen Agenten
        observation, info = env.reset()  # Setze die Umgebung zurück und erhalte die erste Beobachtung
        for _ in range(1000):  # Simuliere bis zu 1000 Schritte
            action = np.argmax(agent.fuehre_aktion(observation))  # Bestimme die Aktion basierend auf der Agentenrichtlinie
            observation, reward, terminated, truncated, info = env.step(action)  # Führe die Aktion aus und erhalte neue Zustände
            agent.reward = reward  # Aktualisiere die Belohnung des Agenten
            # Beende die Episode, wenn das Spiel vorbei ist
            if terminated or truncated:
                break

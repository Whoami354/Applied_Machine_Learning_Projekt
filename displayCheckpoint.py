import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Lade die Datei "rewards.txt" und lese die Werte ein
with open("rewards.txt", "r") as reward_file:
    rewards = reward_file.readlines()
    # Konvertiere die gelesenen Werte in Fließkommazahlen und runde auf 3 Dezimalstellen
    y_values = [np.round(float(reward), decimals=3) for reward in rewards]
    # Erstelle eine Liste von x-Werten entsprechend der Anzahl der rewards
    x_values = range(len(rewards))
    # Plotte die Werte
    plt.plot(x_values, y_values, "-")
    plt.show()

# Erstelle eine Umgebung (environment) für das LunarLander-v2-Spiel mit menschenlesbarer Darstellung
env = gym.make("LunarLander-v2", render_mode='human')
# Lade die Bevölkerung (population) aus der Datei "checkpoint.pickle"
with open("checkpoint.pickle", "rb") as file:
    population = pickle.load(file)
    population_size = len(population)

# Starte die Haupt-Simulationschleife
while True:
    # Iteriere über jeden Agenten in der Population
    for idx in range(population_size):
        agent = population[idx]
        # Setze die Umgebung zurück und erhalte die erste Beobachtung
        observation, info = env.reset()
        # Simuliere das Verhalten des Agenten für bis zu 1000 Schritte
        for _ in range(1000):
            # Wähle die Aktion basierend auf der Agentenrichtlinie (policy)
            action = np.argmax(agent.fuehre_aktion(observation))  # agent policy that uses the observation and info
            # Führe die ausgewählte Aktion aus und erhalte die neuen Zustände
            observation, reward, terminated, truncated, info = env.step(action)
            # Aktualisiere die Belohnung des Agenten
            agent.reward = reward
            # Beende die Schleife, wenn das Spiel beendet oder abgeschnitten ist
            if terminated or truncated:
                agent.reward = reward
                break

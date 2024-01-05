import gymnasium as gym
from GeneticAlgorithm import GA
import numpy as np

# Erstelle eine Umgebung für das Gym (Reinforcement Learning Environment)
env = gym.make("LunarLander-v2")
ga = GA()

# Eine Endlosschleife, die das Training durchführt
while True:
    # Iteriere durch die Population des genetischen Algorithmus
    for idx in range(ga.population_size):
        # Holen Sie sich den Agenten aus der aktuellen Population
        agent = ga.population[idx]
        # Setze die Umgebung zurück und erhalte die erste Beobachtung
        observation, info = env.reset()
        # Iteriere durch maximal 1000 Schritte (Zeitschritte) in der Umgebung
        for _ in range(1000):
            # Wende die Aktion an, die vom Agenten basierend auf der Beobachtung vorgeschlagen wird
            action = np.argmax(agent.fuehre_aktion(observation))  # Agentenrichtlinie, die die Beobachtung und Info verwendet
            observation, reward, terminated, truncated, info = env.step(action)
            # Setze die Belohnung des Agenten basierend auf der aktuellen Belohnung fest
            agent.reward = reward
            # Überprüfe, ob die Episode beendet ist (terminated) oder abgeschnitten (truncated)
            if terminated or truncated:
                break
    # Führe die Reproduktion und Crossover in der genetischen Algorithmus-Population durch
    ga.reproduce_crossover()
    # Speichere den aktuellen Zustand des genetischen Algorithmus (z. B. Gewichte der Agenten)
    ga.checkpoint()
    # Erzwinge die Aktualisierung der Belohnungen in der Population
    ga.force_rewards()
    # Überprüfe, ob der genetische Algorithmus beendet werden soll
    if ga.stop:
        break
# Schließe die Gym-Umgebung
env.close()
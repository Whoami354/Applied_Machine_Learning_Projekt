import gymnasium as gym
from GeneticAlgorithm import GA
import numpy as np

# Erstellen einer neuen Umgebung für das LunarLander-v2 Spiel
env = gym.make("LunarLander-v2")
# Initialisieren des genetischen Algorithmus
ga = GA()
# Starten einer Endlosschleife für das Training
while True:
    # Durchlaufen der gesamten Population der Agenten
    for idx in range(ga.population_size):
        # Auswählen eines Agenten aus der Population
        agent = ga.population[idx]
        # Zurücksetzen der Umgebung zu Beginn jeder Episode und erhalten der Anfangsbeobachtung
        observation, info = env.reset()
        # Durchführen von bis zu 1000 Schritten
        for _ in range(1000):
            # Auswahl der Aktion basierend auf den Beobachtungen und der Strategie des Agenten
            action = np.argmax(agent.fuehre_aktion(observation))
            # Ausführen der Aktion in der Umgebung und Erhalten des neuen Zustands und der Belohnung
            observation, reward, terminated, truncated, info = env.step(action)
            # Aktualisieren der Belohnung des Agenten
            agent.reward = reward
            # Beenden der Episode, wenn das Spiel vorbei ist (entweder erfolgreich gelandet oder abgestürzt)
            if terminated or truncated:
                break
    # Anwenden des genetischen Algorithmus, um die nächste Generation zu erzeugen
    ga.reproduce_crossover()
    # Speichern eines Checkpoints, um den Fortschritt zu speichern
    ga.checkpoint()
    # Erzwingen der Belohnungen, um sicherzustellen, dass alle Belohnungen berücksichtigt werden
    ga.force_rewards()
    # Überprüfen, ob das Trainingskriterium erreicht ist und das Training beenden, wenn ja
    if ga.stop:
        break
# Schließen der Umgebung, wenn das Training beendet ist
env.close()

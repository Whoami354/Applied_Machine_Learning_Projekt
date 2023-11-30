import gymnasium as gym
from Agent import Agent
from GeneticAlgorithm import GA
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")

ga = GA()
while True:

    for idx in range(ga.population_size):
        agent = ga.population[idx]
        observation, info = env.reset()
        for _ in range(1000):
            action = np.argmax(agent.fuehre_aktion(observation)) # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            agent.reward = reward
            if terminated or truncated:
                agent.reward = reward
                break
    ga.reproduce()

env.close()
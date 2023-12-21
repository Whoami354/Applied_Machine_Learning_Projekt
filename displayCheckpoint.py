import pickle

import gymnasium as gym
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
with open("rewards.txt", "r") as reward_file:
    rewards = reward_file.readlines()
    y_values = []
    for reward in rewards:
        y_values.append(np.round(float(reward), decimals=3))
    x_values = range(len(rewards))
    plt.plot(x_values, y_values, "-")
    plt.show()

env = gym.make("LunarLander-v2", render_mode='human')

with open("checkpoint.pickle", "rb") as file:
    population = pickle.load(file)
    population_size = len(population)

while True:

    for idx in range(population_size):
        agent = population[idx]
        observation, info = env.reset()
        for _ in range(1000):
            action = np.argmax(agent.fuehre_aktion(observation))  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            agent.reward = reward
            if terminated or truncated:
                agent.reward = reward
                break
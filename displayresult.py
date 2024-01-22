import gymnasium as gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Loading and plotting rewards from a file
with open("150_008_1200.txt", "r") as reward_file:
    rewards = reward_file.readlines()
    y_values = []
    for reward in rewards:
        # Converts each reward into a floating point number, rounds down to three decimal places
        y_values.append(np.round(float(reward), decimals=3))
    x_values = range(len(rewards))  # Creates a list of x-values according to the number of rewards
    plt.xlabel("Generations")
    plt.ylabel("Rewards")
    plt.plot(x_values, y_values, "-")  # Plot the rewards as a line
    plt.show()

counter = 0
env = gym.make("LunarLander-v2", render_mode='human')
running = True

# Loading the population from a pickle file
with open("checkpoint008.pickle", "rb") as file:
    population = pickle.load(file)  # Loads the population
    population_size = len(population)  # Determines the size of the population

# Starts the main simulation loop
while running:
    # Iterate through the entire population of agents
    for idx in range(population_size):
        # Selecting the current agent from the population
        agent = population[idx]
        # Resets the environment and receives the first observation
        observation, info = env.reset()
        # Simulates up to 1000 steps
        for _ in range(1000):
            # Determines the action based on the agent policy
            action = np.argmax(agent.execute_action(observation))
            # Executes the action and receives new states
            observation, reward, terminated, truncated, info = env.step(action)
            # Updates the agent's reward
            agent.reward = reward
            # End the episode when the game is over (either successfully landed or crashed)
            if terminated or truncated:
                running = False
        # Count up the number of rewards and print them out
        counter += 1
        print(f"Reward {counter}: {agent.reward}")

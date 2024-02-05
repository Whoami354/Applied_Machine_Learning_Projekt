import gymnasium as gym
from GeneticAlgorithm import GA
import numpy as np

env = gym.make("LunarLander-v2")
ga = GA()
running = True

# Start an endless loop for training
while running:
    # Iterate through the entire population of agents
    for idx in range(ga.population_size):
        # Selecting the current agent from the population
        agent = ga.population[idx]
        # Reset the environment at the beginning of each episode and maintain the initial observation
        observation, info = env.reset()
        # Perform up to 1000 steps
        for _ in range(1000):
            # Selection of the action based on the agent's observations and strategy
            action = np.argmax(agent.execute_action(observation))
            # Perform the action in the environment and receive the new status and reward
            observation, reward, terminated, truncated, info = env.step(action)
            # Updating the agent's reward
            agent.reward = reward
            # End the episode when the game is over (either successfully landed or crashed)
            if terminated or truncated:
                break
    # Applying the genetic algorithm to create the next generation
    ga.reproduce_crossover()
    # Saving a checkpoint to save the current generation of agents
    ga.checkpoint()
    # The average reward is written to a file
    ga.force_rewards()
    # Check whether the training criterion has been reached and end the training
    if ga.stop:
        running = False
# Close the environment when training is finished
env.close()

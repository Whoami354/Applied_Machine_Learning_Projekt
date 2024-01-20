import pickle
import random

from Agent import Agent as ag
import numpy as np

# Definition of the class GA (Genetic Algorithm)
class GA:
    def __init__(self):
        # Initialization of the properties of the genetic algorithm
        self.goal = 150  # Termination condition
        self.mutation_rate = 0.08  # Mutation probability
        self.population = []  # List of agents (population)
        self.average_reward = []  # List of average rewards
        self.generation_number = 0  # Current generation
        self.stop = False  # Termination condition (flag)
        self.population_size = 1200  # Size of the population

        # Initialization of the starting population
        for idx in range(self.population_size):
            self.population.append(ag())  # Adds new agents to the population

    def mutation(self, a):
        # Method for performing a mutation on an agent
        agent = a.clone()  # Creates a copy of the agent
        for idx in range(len(agent.brain.weights)):
            matrix = agent.brain.weights[idx]  # Access to the weight matrix
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rand = np.random.uniform()  # Generates a random number
                    # Applies mutation if the random number is smaller than the mutation rate
                    if rand < self.mutation_rate:
                        matrix[i, j] += np.random.uniform(-0.015, 0.015)  # Changes the weight slightly
        return agent  # Returns the mutated agent

    def reproduce_crossover(self):
        # Population reproduction method with crossover
        print(self.generation_number)
        sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
        self.average_rewards(self.goal)  # Updates the rewards and checks the termination condition
        n = int(len(sorted_agents) * 0.1)  # Determines the number of top 10%
        top_10 = sorted_agents[:n]  # Selects the best 10%
        new_population = []
        for idx in range(self.population_size):
            random_first_parent = top_10[random.randrange(n)] # Randomly chooses the first parent
            random_second_parent = top_10[random.randrange(n)] # Randomly chooses the second parent
            child = self.crossover(random_first_parent, random_second_parent)  # Creates a child through crossover
            child = self.mutation(child)  # Applies mutation to the child
            new_population.append(child)  # Adds the child to the new population
        self.population = new_population  # Updates the population
        self.generation_number += 1  # Increases the number of generations

    def average_rewards(self, threshold):
        # Method for calculating the average reward and checking the termination condition
        rewards = sum(map(lambda agent: agent.reward, self.population))
        average_reward = rewards / len(self.population)
        self.average_reward.append(average_reward)
        if average_reward > threshold:  # Checks whether the average reward has exceeded the target
            self.stop = True  # Sets the cancel flag
            self.force_rewards()  # Write the rewards to a file
        print("rewards:", self.average_reward[-1])

    def force_rewards(self):
        # Method for writing the average rewards to a file
        with open("rewards.txt", "w") as file:
            for line in self.average_reward:
                file.write(str(line) + "\n")  # Writes each reward to the file

    def crossover(self, first_partent, second_parent):
        # Method for performing a crossover between two agents
        child = ag()  # Creates a new agent (child)
        parents = [first_partent, second_parent]  # Defines the parents
        for idx in range(len(child.brain.weights)):
            matrix = child.brain.weights[idx]  # Access to the child's weight matrix
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    parent = parents[random.randrange(2)]  # Chooses a parent at random
                    parent_value = parent.brain.weights[idx][i, j]  # Get the appropriate weight
                    matrix[i, j] = parent_value  # Sets the weight of the parent
        return child  # Returns the child

    def checkpoint(self):
        # Method for saving the current population in a file
        with open("checkpoint008_v2.pickle","wb") as file:
            pickle.dump(self.population, file)  # Writes the population to the file

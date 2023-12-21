import pickle
import random

from Agent import Agent as ag
import numpy as np

class GA:
    def __init__(self):
        self.Goal = 150
        self.mutation_rate = 0.10
        self.population = []
        self.average_reward = []
        self.generation_number = 0
        self.stop = False
        self.population_size = 1200

        for idx in range(self.population_size):
            self.population.append(ag())

    def reproduce(self):
        print(self.generation_number)
        sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
        # Die besten 20%:
        rewards = sum(map(lambda agent: agent.reward, sorted_agents))
        print("rewards:", rewards / len(self.population))
        n = int(len(sorted_agents) * 0.2)
        top_20 = sorted_agents[:n]
        new_population = []
        for idx in range(self.population_size):
            candidate = top_20[idx % n]
            child = self.mutation(candidate)
            new_population.append(child)
            # die besten 20% wollen wir auswählen und diese miteinander kreuzen und rotieren
        self.generation_number += 1

        self.population = new_population

    def mutation(self, a):
        agent = a.clone()
        for idx in range(len(agent.brain.weights)):
            matrix = agent.brain.weights[idx]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rand = np.random.uniform()
                    if rand < self.mutation_rate:
                        matrix[i, j] += np.random.uniform(-0.015, 0.015)
        return agent

    def reproduce_crossover(self):
        print(self.generation_number)
        sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
        # Die besten 20%:
        self.get_rewards(self.Goal)
        n = int(len(sorted_agents) * 0.1)
        top_20 = sorted_agents[:n]
        new_population = []
        for idx in range(self.population_size):
            candidate1 = top_20[random.randrange(n)]
            candidate2 = top_20[random.randrange(n)]
            child = self.crossover(candidate1, candidate2)
            child = self.mutation(child)
            new_population.append(child)
        self.population = new_population
        self.generation_number += 1

    def get_rewards(self, threshold):
        rewards = sum(map(lambda agent: agent.reward, self.population))
        average_reward = rewards / len(self.population)
        self.average_reward.append(average_reward)
        if average_reward > threshold:
            self.stop = True
            self.force_rewards()
        print("rewards:", self.average_reward[-1])

    def force_rewards(self):
        with open("rewards.txt", "w") as file:
            for line in self.average_reward:
                file.write(str(line) + "\n")

    def crossover(self, candidate1, candidate2):
        child = ag()
        parents = [candidate1, candidate2]
        for idx in range(len(child.brain.weights)):
            matrix = child.brain.weights[idx]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    parent = parents[random.randrange(2)]
                    parent_value = parent.brain.weights[idx][i, j]
                    matrix[i, j] = parent_value
        return child

    def checkpoint(self):
        with open("checkpoint.pickle","wb") as file:
            pickle.dump(self.population, file)

from Agent import Agent as ag
import numpy as np

class GA:
    def __init__(self):
        self.mutation_rate = 0.1
        self.population = []

        self.population_size = 5

        for idx in range(self.population_size):
            self.population.append(ag())

    def reproduce(self):
        sorted_agents = sorted(self.population, key=lambda x: x.reward, reverse=True)
        for agents in sorted_agents:
            print(agents.reward)
            self.mutation(agents)
            # die besten 20% wollen wir auswählen und diese miteinander kreuzen und rotieren

        self.population = sorted_agents

    def mutation(self, agent):
        for idx in range(len(agent.dna)):
            matrix = agent.dna[idx]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rand = np.random.uniform()
                    if rand < self.mutation_rate:
                        matrix[i,j] += np.random.uniform(-0.1, 0.1)






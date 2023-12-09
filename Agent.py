from NeuronalNetwork import NeuronalNetwork as nn

class Agent:
    def __init__(self):
        self.brain = nn([8, 4])
        self.dna = self.brain.weights
        self.reward = 0

    def fuehre_aktion(self, input_vector):
        return self.brain.feet_forward(input_vector)

    def __str__(self):
        return str(self.reward)

    def clone(self):
        child = Agent()
        child.brain = self.brain.copy()
        child.dna = child.brain.weights
        return child

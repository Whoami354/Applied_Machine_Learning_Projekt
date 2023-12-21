from NeuronalNetwork import NeuronalNetwork as nn

class Agent:
    def __init__(self):
        # jeder Agent hat Gehirn, um entscheidungen zu treffen, basierend auf den Observations (was er beobachtet)
        self.brain = nn([8, 4])
        # das ist die Belohnung, was man gibt, wenn er die Arbeit gut gemacht hat.
        self.reward = 0

    def fuehre_aktion(self, input_vector):
        return self.brain.feed_forward(input_vector)

    def __str__(self):
        return str(self.reward)

    def clone(self):
        child = Agent()
        child.brain = self.brain.copy()
        return child

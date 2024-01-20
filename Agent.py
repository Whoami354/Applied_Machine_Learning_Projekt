from NeuronalNetwork import NeuronalNetwork as nn

# Definition of the Agent class
class Agent:
    def __init__(self):
        # The neural network has 8 input neurons and 4 output neurons
        """
        8 input eurons:
        X-distance from target
        Y-distance from target
        X-speed
        Y-speed
        Inclination of the ship
        Angular speed of the ship
        Left leg on the ground (True/False)
        Right leg on the ground (True/False)
        """
        self.brain = nn([8, 4])
        # Reward is used as a value for performance
        self.reward = 0

    def execute_action(self, input_vector):
        # Method to perform an action based on a given input vector
        # The method uses the feed_forward function of the neural network to determine the action
        return self.brain.feed_forward(input_vector)

    def __str__(self):
        return str(self.reward)

    def clone(self):
        # Method to create a copy (clone) of the agent
        child = Agent()  # Creates a new agent
        child.brain = self.brain.copy()  # Copies the brain (neural network) of the current agent into the new agent
        return child  # Returns the new agent

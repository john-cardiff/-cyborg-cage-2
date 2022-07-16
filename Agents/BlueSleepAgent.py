from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

class BlueSleepAgent(BaseAgent):
    def __init__(self):
        # action 0 is sleep
        self.action = 0

    # unecessary to add observation but need it to be consistent with our other agents
    def get_action(self, observation, action_space=None):
        return self.action

    def train(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

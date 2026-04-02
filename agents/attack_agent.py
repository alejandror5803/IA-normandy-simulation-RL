
class attack_Agent:
    def __init__(self, peloton, team):
        self.peloton = peloton
        self.team = team
        
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.9

        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        # acciones: 0: no atacar, 1: atacar.
        self.action_space = [0, 1]
        
        self.qtable = {}


    def act(self, state):
        pass
    def observe(self, state):
        pass
    def reset(self):
        pass

    def take_decision(self, state):
        pass

    
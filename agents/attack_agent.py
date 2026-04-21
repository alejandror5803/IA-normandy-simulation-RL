# implementing the libraries
import numpy as np
import random
import agents.agents_config as cfg
 
# actions
DONT_SHOOT = cfg.DONT_SHOOT
SHOOT = cfg.SHOOT

class attack_agent:
    
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        """
        Inizialate the Attack Agent

        Defining the parameters, like the Leraning Rate, gamma & epsilon.
        All as a float. Including the q table values.

        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
 
        # state: 0 = no enemy in range, 1 = enemy in range
        # actions: 0 = dont shoot, 1 = shoot
        self.q_table = np.zeros((2, 2))
 
    def get_state(self, enemies_in_range):

        """
        Function gets the state of the enemies if they are near.

        Parameters:
        enemies_in_range: It gives you an alert of nearness.

        Return:
            It returns in form of int 0 or 1, depending of the enemy.

        """
        if len(enemies_in_range) > 0:
            return 1
        return 0
 
    def choose_action(self, state):
        """
        Chooses each action deppending of the q table

        Parameters:
            state: in the q table

        Return:
            It returns an int from the index of the selected action
        """

        if random.random() < self.epsilon:
            return random.randint(0, 1)
        

        return int(np.argmax(self.q_table[state]))
 
    def compute_reward(self, state, action, hit_confirmed=False):
        """
        Quantificates an reward depending on the decision

        Parameters:
            State : of the agent.
            action : what has the agent decided
            hit confirmated: Has he hit the enemy? 
        

        Return:
           The reward based on the action of the Agent.
        
        """
        if state == 1:  # there was an enemy
            if action == SHOOT:
                reward = 1.5 if hit_confirmed else 0.5
            else:
                reward = -1.0  # enemy visible and didnt shoot
        else:
            reward = 0.0  # no enemy, neutral
 
        return reward
 
    def update(self, state, action, reward, next_state):
        """
        Updates the state of the Agent

        Parameters:
        state: Current state of the Agent
        action: latest action of the Agent
        Next_state: The best move for the Agent 
        
        """
        best_next_q = np.max(self.q_table[next_state]) # Chooses the best next step in the q table
        old_q = self.q_table[state][action] # Saves the recent state
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q) # Calculates the state and action
 
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Chooses the maximum epsilon between the min epsilon 
        and the current * the decay rate

        Parameters:
        decay_rate=0.995 
        min_epsilon=0.05

        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
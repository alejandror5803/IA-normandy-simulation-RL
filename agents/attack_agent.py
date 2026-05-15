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
        Initializes the Attack Agent.

        Defining the parameters, like the Learning Rate, gamma & epsilon.
        All as a float. Including the Q-table values.
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
 
        # state: 0 = no enemy in range, 1 = enemy in range
        # actions: 0 = dont shoot, 1 = shoot
        self.q_table = np.zeros((2, 2))
 
    def get_state(self, enemies_in_range):
        """
        Returns the attack state: 0 if no enemies in range, 1 if at least one is.

        Parameters:
        enemies_in_range: list of enemy platoons within attack range.

        Return:
            0 or 1 as an integer.
        """
        if len(enemies_in_range) > 0:
            return 1
        return 0
 
    def choose_action(self, state):
        """
        Chooses an action depending on epsilon-greedy policy over the Q-table.

        Parameters:
            state: current state index in the Q-table.

        Return:
            An integer index of the selected action.
        """

        if random.random() < self.epsilon:
            return random.randint(0, 1)
        

        return int(np.argmax(self.q_table[state]))
 
    def compute_reward(self, state, action, hit_confirmed=False):
        """
        Calculates the reward depending on the decision taken.

        Parameters:
            state: current state of the agent.
            action: action chosen by the agent.
            hit_confirmed: whether the shot actually hit an enemy.

        Return:
           The reward based on the action of the agent.
        """
        if state == 1:  # there was an enemy
            if action == SHOOT:
                reward = 1.5 if hit_confirmed else 0.5
            else:
                reward = -1.0  # enemy visible and didnt shoot
        else:
            # small penalty: shooting blindly should not look as good as waiting
            reward = -0.3 if action == SHOOT else 0.0
 
        return reward
 
    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table with a Bellman step.

        Parameters:
        state: current state of the agent.
        action: latest action of the agent.
        reward: reward received.
        next_state: state observed after taking the action.
        """
        best_next_q = np.max(self.q_table[next_state]) # Chooses the best next step in the q table
        old_q = self.q_table[state][action] # Saves the recent state
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q) # Calculates the state and action
 
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Decays epsilon by decay_rate, clamped to min_epsilon.

        Parameters:
        decay_rate: multiplicative decay factor (default 0.995).
        min_epsilon: lower bound for epsilon (default 0.05).
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
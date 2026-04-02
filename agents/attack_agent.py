import numpy as np
import random
 
 
# actions
DONT_SHOOT = 0
SHOOT = 1

class attack_agent:
 
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
 
        # state: 0 = no enemy in range, 1 = enemy in range
        # actions: 0 = dont shoot, 1 = shoot
        self.q_table = np.zeros((2, 2))
 
    def get_state(self, enemies_in_range):
        if len(enemies_in_range) > 0:
            return 1
        return 0
 
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_table[state]))
 
    def compute_reward(self, state, action, hit_confirmed=False):
        if state == 1:  # there was an enemy
            if action == SHOOT:
                reward = 1.5 if hit_confirmed else 0.5
            else:
                reward = -1.0  # enemy visible and didnt shoot
        else:
            reward = 0.0  # no enemy, neutral
 
        return reward
 
    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q)
 
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
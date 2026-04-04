import numpy as np
import random


# cover types (only 3 to test)
NO_COVER = 0
BUSH = 1
WALL = 2

# actions
DONT_COVER = 0
TAKE_COVER = 1

# how much each cover type helps
COVER_BONUS = {NO_COVER: 0.0, BUSH: 0.5, WALL: 1.0}


class defense_agent:

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # state: (enemy_nearby: 0/1) x (cover_type: 0/1/2) = 6 states
        # actions: 0 = stay in open, 1 = take cover
        num_states = 6
        num_actions = 2
        self.q_table = np.zeros((num_states, num_actions))

    def _state_index(self, enemy_nearby, cover_type):
        return int(enemy_nearby) * 3 + cover_type

    def choose_action(self, enemy_nearby, cover_type):
        state = self._state_index(enemy_nearby, cover_type)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_table[state]))

    def compute_reward(self, enemy_nearby, cover_type, action, got_hit):
        reward = 0.0

        if enemy_nearby:
            if action == TAKE_COVER:
                # more reward for better cover
                reward += 0.5 + COVER_BONUS[cover_type]
            else:
                reward -= 1.0  # enemy nearby and exposed, bad

        if got_hit:
            if action == TAKE_COVER:
                # got hit but was covered, partial mitigation
                reward += 0.3 * COVER_BONUS.get(cover_type, 0)
            else:
                # got hit in the open, really bad
                reward -= 2.5

        return reward

    def update(self, enemy_nearby, cover_type, action, reward, next_enemy_nearby, next_cover_type):
        state = self._state_index(enemy_nearby, cover_type)
        next_state = self._state_index(next_enemy_nearby, next_cover_type)

        best_next_q = np.max(self.q_table[next_state])
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
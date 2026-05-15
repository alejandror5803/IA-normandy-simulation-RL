import numpy as np
import random
import agents.agents_config as cfg

# cover types (only 3 to test)
NO_COVER = cfg.NO_COVER
BUSH = cfg.BUSH
WALL = cfg.WALL

# actions
DONT_COVER = cfg.DONT_COVER
TAKE_COVER = cfg.TAKE_COVER

# how much each cover type helps
COVER_BONUS = cfg.COVER_BONUS


class defense_agent:

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        """
        Initializes the Defense Agent.

        Defining the parameters, like the Learning Rate, gamma & epsilon.
        All as a float. As well as the number of states and actions possible for the Q-table.
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # state: (enemy_nearby: 0/1) x (cover_type: 0/1/2) = 6 states
        # actions: 0 = stay in open, 1 = take cover
        num_states = 6
        num_actions = 2
        self.q_table = np.zeros((num_states, num_actions))

    # Gives an int for the state
    def _state_index(self, enemy_nearby, cover_type):
        return int(enemy_nearby) * 3 + cover_type
    
     # Chooses best action
    def choose_action(self, enemy_nearby, cover_type):
        state = self._state_index(enemy_nearby, cover_type)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_table[state]))

    def compute_reward(self, enemy_nearby, cover_type, action, got_hit, next_cover_type=0):
        """
        Calculates the reward depending on the decision taken.

        Parameters:
        enemy_nearby: whether an enemy is within nearby range (0 or 1).
        cover_type: current cover type at the platoon's position.
        action: latest action of the agent.
        got_hit: whether the platoon took damage this step.
        next_cover_type: cover type after the action was taken.

        Return:
           The reward based on the action of the agent.
        """
        
        
        reward = 0.0

        if enemy_nearby:
            if action == TAKE_COVER:
                if next_cover_type > cover_type:
                    reward += 1.0   # actually moved to better cover — good
                elif cover_type > 0:
                    reward += 0.1   # already in cover and staying — acceptable but not great
                else:
                    reward -= 0.5   # tried to get cover but no better cell adjacent
            else:
                if cover_type == 0:
                    reward -= 1.0   # enemy nearby, in the open, not even trying to cover

        if got_hit:
            if cover_type > 0:
                reward += 0.3 * COVER_BONUS[cover_type]   # cover absorbed some damage
            else:
                reward -= 2.5   # took damage fully in the open

        return reward

    def update(self, enemy_nearby, cover_type, action, reward, next_enemy_nearby, next_cover_type):
        """
        Updates the Q-table with a Bellman step.

        Parameters:
        enemy_nearby: whether an enemy was nearby before the action.
        cover_type: cover type before the action.
        action: latest action of the agent.
        reward: reward received.
        next_enemy_nearby: whether an enemy is nearby after the action.
        next_cover_type: cover type after the action.
        """
        state = self._state_index(enemy_nearby, cover_type)
        next_state = self._state_index(next_enemy_nearby, next_cover_type)

        best_next_q = np.max(self.q_table[next_state])
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Decays epsilon by decay_rate, clamped to min_epsilon.

        Parameters:
        decay_rate: multiplicative decay factor (default 0.995).
        min_epsilon: lower bound for epsilon (default 0.05).
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
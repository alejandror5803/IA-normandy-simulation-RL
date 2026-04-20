# implementing the libraries
import numpy as np
import random
import agents.agents_config as cfg

# actions
MOVE_UP = cfg.MOVE_UP
MOVE_DOWN = cfg.MOVE_DOWN
MOVE_LEFT = cfg.MOVE_LEFT
MOVE_RIGHT = cfg.MOVE_RIGHT
STAY = cfg.STAY

# Defines the class
class capture_agent:
    """
    Inizialites the class o the capture agent
    
     Defining the parameters, like the Leraning Rate, gamma & epsilon.
        All as a float. Also defining the actions and states possible inizialising the q table.
    """
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.3):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # state: direction to objective (dx_sign, dy_sign) + distance bucket
        # dx_sign: 0=same col, 1=obj is right, 2=obj is left
        # dy_sign: 0=same row, 1=obj is below, 2=obj is above
        # dist_bucket: 0=close(<3 tiles), 1=medium(3-7), 2=far(>7)
        # total: 3 * 3 * 3 = 27 states
        num_states = 27
        num_actions = 5
        self.q_table = np.zeros((num_states, num_actions))

        # to detect oscillation (going back and forth)
        self.prev_position = None
        self.prev_prev_position = None

    def _get_state_index(self, agent_pos, objective_pos):

        """
        It gives the capture agent where to direct himself so he can go to the capture point.
        
        Parameters:
        Agent_pos: Where he is at the moment
        objectice_pos: Where the objective is

        Return the value of in which direction is that agent in respect of the Agent
        """
        dx = objective_pos[0] - agent_pos[0]
        dy = objective_pos[1] - agent_pos[1]

        if dx == 0:
            dx_sign = 0
        elif dx > 0:
            dx_sign = 1  # objective is to the right
        else:
            dx_sign = 2  # objective is to the left

        if dy == 0:
            dy_sign = 0
        elif dy > 0:
            dy_sign = 1  # objective is below
        else:
            dy_sign = 2  # objective is above

        manhattan_dist = abs(dx) + abs(dy)
        if manhattan_dist < 3:
            dist_bucket = 0
        elif manhattan_dist <= 7:
            dist_bucket = 1
        else:
            dist_bucket = 2

        return dx_sign * 9 + dy_sign * 3 + dist_bucket

    def choose_action(self, agent_pos, objective_pos):
        """
        Chooses each action deppending of the q table

        Parameters:
        Agent_pos: Where he is at the moment
        objectice_pos: Where the objective is

        Return:
            It returns an int from the index of the selected action
        """
         
        state = self._get_state_index(agent_pos, objective_pos)
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return int(np.argmax(self.q_table[state]))

    def compute_reward(self, old_pos, new_pos, objective_pos):
        """
        Quantificates an reward depending on the decision

        Parameters:
        old_pos: last position
        new_pos: nex position
        objectice_pos: Where the objective is

        Return:
           The reward based on the action of the Agent.
        
        """
        old_dist = abs(objective_pos[0] - old_pos[0]) + abs(objective_pos[1] - old_pos[1])
        new_dist = abs(objective_pos[0] - new_pos[0]) + abs(objective_pos[1] - new_pos[1])

        reward = 0.0

        if new_dist < old_dist:
            reward += 1.0  # getting closer to objective
        elif new_dist > old_dist:
            reward -= 1.5  # moving away from objective

        # penalize going back to where we were two steps ago (oscillation)
        if self.prev_prev_position is not None:
            if new_pos == self.prev_prev_position:
                reward -= 2.0

        # objective captured
        if new_dist == 0:
            reward += 10.0

        return reward

    # Updates the position of the capture agent
    def update_position_history(self, new_pos):

        self.prev_prev_position = self.prev_position
        self.prev_position = new_pos

    # History of position to None
    def reset_position_history(self):
        
        # call this when objective changes
        self.prev_position = None
        self.prev_prev_position = None


    def update(self, agent_pos, objective_pos, action, reward, new_agent_pos):
        """
        Updates the state of the Agent

        Parameters:
        Agent_pos: Where he is at the moment
        objectice_pos: Where the objective is
        action: latest action of the Agent
        reward: quantitative value
        New_agent_pos: How is positionated.
        
        """
        state = self._get_state_index(agent_pos, objective_pos)
        next_state = self._get_state_index(new_agent_pos, objective_pos)

        best_next_q = np.max(self.q_table[next_state])
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.lr * (reward + self.gamma * best_next_q - old_q)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Chooses the maximum epsilon between the min epsilon 
        and the current * the decay rate

        Parameters:
        decay_rate=0.995 
        min_epsilon=0.05

        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
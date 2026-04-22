import numpy as np
import random
import agents.agents_config as cfg

# mirrors the meta-action IDs in normandy_env.py
META_CAPTURE  = cfg.META_CAPTURE
META_ATTACK   = cfg.META_ATTACK
META_DEFENSE  = cfg.META_DEFENSE
META_RESUPPLY = cfg.META_RESUPPLY

# same as ATTACK_RANGE in combat_logic.py
ATTACK_RANGE = cfg.ATTACK_RANGE


class command_agent:

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.6): # 0.3 epsilon inicial para fomentar exploración, luego se va decayendo
        """
        Inizialites the class o the capture agent
    
        Defining the parameters, like the Leraning Rate, gamma & epsilon.
        All as a float. As well as the number of states and actions possibles for the q table
        """
        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon

        # state per peloton is built from the 16-value obs vector.
        # key change from before: enemy_nearby (2 values) is replaced by
        # threat_level (3 values) which tells us HOW close the enemy is:
        #
        #   hp_level      : 0=low(0-1), 1=mid(2-3), 2=high(4-5)       -> 3 values
        #
        #   threat_level  : 0 = no enemy nearby (obs[5] == 0)
        #                   1 = enemy nearby but out of attack range
        #                       (obs[5]==1, obs[6] > ATTACK_RANGE)
        #                   2 = enemy IN attack range
        #                       (obs[5]==1, obs[6] <= ATTACK_RANGE)    -> 3 values
        #
        #   obj_dx_dir    : obs[10], 0=same, 1=right, 2=left           -> 3 values
        #   obj_dy_dir    : obs[11], 0=same, 1=below, 2=above          -> 3 values
        #   low_ammo      : obs[15], 0 or 1                            -> 2 values
        #
        # total: 3 * 3 * 3 * 3 * 2 = 162 states
        # actions: 4 (META_CAPTURE, META_ATTACK, META_DEFENSE, META_RESUPPLY)

        num_states  = 162
        num_actions = 4
        self.q_table = np.zeros((num_states, num_actions))

    # Depending on the observation of the peloton it will change
    def _obs_to_state(self, peloton_obs):
        hp_bucket    = int(peloton_obs[0])
        enemy_nearby = int(peloton_obs[5])
        enemy_dist   = int(peloton_obs[6])
        obj_dx_dir   = int(peloton_obs[10])
        obj_dy_dir   = int(peloton_obs[11])
        low_ammo     = int(peloton_obs[15])

        if hp_bucket <= 1:
            hp_level = 0
        elif hp_bucket <= 3:
            hp_level = 1
        else:
            hp_level = 2

        if enemy_nearby == 0:
            threat_level = 0   # no enemy nearby at all
        elif enemy_dist > ATTACK_RANGE:
            threat_level = 1   # enemy nearby but can't shoot yet
        else:
            threat_level = 2   # enemy is literally in shooting range

        # encoding: hp(3) * threat(3) * dx(3) * dy(3) * ammo(2)
        state = (hp_level * 54) + (threat_level * 18) + (obj_dx_dir * 6) + (obj_dy_dir * 2) + low_ammo
        return state

    # Chooses action only from the peloton
    def choose_action(self, peloton_obs):
        state = self._obs_to_state(peloton_obs)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

    # Chooses best action for the whole team
    def choose_actions_for_team(self, obs_tuple):
        actions = []
        for peloton_obs in obs_tuple:
            actions.append(self.choose_action(peloton_obs))
        return actions

    def compute_reward(self, obs_before, obs_after, action, env_reward):
        """
        Quantificates an reward depending on the decision

        Parameters:
        obs_before: observation at the current state
        obs_after: observation after the current state
        action: latest action of the Agent
        reward: quantitative value
        env_reward = The reward in each iteration

        Return:
           The reward based on the action of the Agent.
        
        """

        reward = env_reward

        enemy_nearby = int(obs_before[5])
        enemy_dist   = int(obs_before[6])
        in_atk_range = (enemy_nearby == 1 and enemy_dist <= ATTACK_RANGE)

        # penalize heavily: chose to go capture when an enemy was right there to shoot
        if action == META_CAPTURE and in_atk_range:
            reward -= 3.0

        # penalize lightly: chose attack mode when there was nothing to attack
        if action == META_ATTACK and enemy_nearby == 0:
            reward -= 0.5

        # penalize: chose defense when neither enemy nearby nor low hp
        hp_before = int(obs_before[0])
        if action == META_DEFENSE and enemy_nearby == 0 and hp_before >= 4:
            reward -= 0.3

        # existing shaping
        if int(obs_after[0]) < hp_before:
            reward -= 0.5   # took damage this step

        obj_dist_before = int(obs_before[12])
        obj_dist_after  = int(obs_after[12])
        if obj_dist_after < obj_dist_before:
            reward += 0.3
        elif obj_dist_after > obj_dist_before:
            reward -= 0.2

        return reward

    def update(self, obs_before, action, reward, obs_after):
        """
        Updates the state of the Agent

        Parameters:
        
        obs_before: observation at the current state
        action: latest action of the Agent
        reward: quantitative value
        obs_after: observation after the current state
        
        """
        state      = self._obs_to_state(obs_before)
        next_state = self._obs_to_state(obs_after)

        best_next_q = np.max(self.q_table[next_state])
        old_q       = self.q_table[state][action]
        td_error    = reward + self.gamma * best_next_q - old_q
        self.q_table[state][action] = old_q + self.lr * td_error
        return abs(td_error)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Chooses the maximum epsilon between the min epsilon 
        and the current * the decay rate

        Parameters:
        decay_rate=0.995 
        min_epsilon=0.05

        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

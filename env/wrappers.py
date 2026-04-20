#implements 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.env_config import OBS_SIZE, NUM_BLUE

# max possible value for each of the 16 obs features — used by ObsNormWrapper
OBS_MAX_VALS = np.array([
    5, 5, 5, 5,   # hp_hundreds, fuel_level, ammo_level, num_tanks
    2, 1, 9,      # cover_type, enemy_nearby, enemy_dist
    1, 1, 1,      # captured_A, captured_B, captured_C
    2, 2, 9,      # obj_dx_dir, obj_dy_dir, obj_dist
    4, 4,         # sector_x, sector_y
    1,            # low_ammo_flag
], dtype=np.float32)


class ObsNormWrapper(gym.ObservationWrapper):
    # Normalizes every obs feature to [0, 1] by dividing by its known max value.
    # Not used during Q-table training (int() bucketing would lose info),
    # but ready to drop in if you ever switch to a neural network policy.

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for _ in range(NUM_BLUE)
        ])

    def observation(self, obs):
        safe_max = np.where(OBS_MAX_VALS == 0, 1.0, OBS_MAX_VALS)
        normalized = []
        for pel_obs in obs:
            normalized.append(np.clip(pel_obs / safe_max, 0.0, 1.0))
        return normalized


class EpisodeStatsWrapper(gym.Wrapper):
    # Keeps a rolling window of per-episode stats and injects them into info
    # whenever an episode ends. Useful to monitor training without matplotlib.
    # Call get_stats() at any point to read the current averages.

    def __init__(self, env, window=100):
        super().__init__(env)
        self.window = window
        self.ep_rewards   = []
        self.ep_steps     = []
        self.ep_captures  = []
        self._ep_reward   = 0.0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)
        self._ep_reward += sum(rewards)

        if terminated or truncated:
            caps_this_ep = sum(1 for v in info['captured'].values() if v)
            self.ep_rewards.append(self._ep_reward)
            self.ep_steps.append(info['step'])
            self.ep_captures.append(caps_this_ep)

            if len(self.ep_rewards) > self.window:
                self.ep_rewards.pop(0)
                self.ep_steps.pop(0)
                self.ep_captures.pop(0)

            n = len(self.ep_rewards)
            info['stats_avg_reward']   = sum(self.ep_rewards)  / n
            info['stats_avg_steps']    = sum(self.ep_steps)    / n
            info['stats_avg_captures'] = sum(self.ep_captures) / n
            info['stats_total_eps']    = n

        return obs, rewards, terminated, truncated, info

    def get_stats(self):
        if not self.ep_rewards:
            return {}
        n = len(self.ep_rewards)
        return {
            'avg_reward':   sum(self.ep_rewards)  / n,
            'avg_steps':    sum(self.ep_steps)    / n,
            'avg_captures': sum(self.ep_captures) / n,
            'total_eps':    n,
        }


class FogOfWarWrapper(gym.ObservationWrapper):
    # Applies fog of war: if the nearest enemy is further than `visibility_range`
    # cells, the enemy features in the obs vector are hidden (zeroed out).
    # Historically grounded — WWII units had limited battlefield visibility,
    # so the agent has to act without knowing where distant enemies are.

    ENEMY_NEARBY_IDX = 5
    ENEMY_DIST_IDX   = 6

    def __init__(self, env, visibility_range=8):
        super().__init__(env)
        self.visibility_range = visibility_range

    def observation(self, obs):
        result = []
        for pel_obs in obs:
            masked = pel_obs.copy()
            enemy_dist = int(pel_obs[self.ENEMY_DIST_IDX])
            if enemy_dist > self.visibility_range:
                masked[self.ENEMY_NEARBY_IDX] = 0.0
                masked[self.ENEMY_DIST_IDX]   = 9.0
            result.append(masked)
        return result


class ActionMaskWrapper(gym.Wrapper):
    # Computes which meta-actions are actually valid for each peloton right now
    # and exposes that as 'action_masks' in info. Also fixes invalid actions before
    # they reach the env, so the agent never wastes a step on something impossible.
    #
    # Rules:
    #   META_ATTACK   (1) -> invalid if peloton has no ammo
    #   META_RESUPPLY (3) -> invalid if peloton is not standing on a supply point

    META_ATTACK   = 1
    META_RESUPPLY = 3
    FALLBACK      = 0  # META_CAPTURE — always a safe default

    def __init__(self, env, redirect_invalid=True):
        super().__init__(env)
        self.redirect_invalid = redirect_invalid

    def _compute_masks(self):
        base_env = self.unwrapped
        masks = []
        for pel in base_env.blue_pelotons:
            mask = [True, True, True, True]

            if pel['num_tanks'] <= 0:
                masks.append(mask)
                continue

            if pel['ammo'] <= 0:
                mask[self.META_ATTACK] = False

            on_supply_point = any(
                pel['pos'] == pd['pos']
                for pd in base_env.points.values()
            )
            if not on_supply_point:
                mask[self.META_RESUPPLY] = False

            masks.append(mask)
        return masks

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['action_masks'] = self._compute_masks()
        return obs, info

    def step(self, actions):
        if self.redirect_invalid:
            masks = self._compute_masks()
            corrected_actions = list(actions)
            for i, (action, mask) in enumerate(zip(actions, masks)):
                if not mask[action]:
                    corrected_actions[i] = self.FALLBACK
            actions = corrected_actions

        obs, rewards, terminated, truncated, info = self.env.step(actions)
        info['action_masks'] = self._compute_masks()
        return obs, rewards, terminated, truncated, info

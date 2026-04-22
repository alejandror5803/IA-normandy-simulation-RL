
#Implementation of the librarys and imports from other modules their values
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
import random
import env.env_config as efg

from env.map_generator import MapGenerator, FIXED_POINTS, POINT_SUPPLY_LIMITS, IMPASSABLE
from env.combat_logic import (
    get_enemies_in_range, get_nearest_enemy, do_attack,
    get_best_cover_cell, get_cover_type_int, do_resupply,
    all_dead, ATTACK_RANGE, TIGER_DAMAGE
)
from env.units import distance

from agents.attack_agent   import attack_agent,  SHOOT
from agents.defense_agent  import defense_agent, TAKE_COVER
from agents.capture_agent  import capture_agent, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY
from agents.command_agent  import command_agent

NUM_BLUE = efg.NUM_BLUE
NUM_RED  = efg.NUM_RED
MAP_SIZE = efg.MAP_SIZE


# commander meta-actions: the commander decides WHICH sub-agent takes action on the peloton this step
# movement direction is never chosen by the commander directly — the capture_agent handles that
META_CAPTURE  = efg.META_CAPTURE   # capture_agent d ecides where to move
META_ATTACK   = efg.META_ATTACK   # attack_agent decides whether to shoot
META_DEFENSE  = efg.META_DEFENSE   # defense_agent decides whether to seek cover
META_RESUPPLY = efg.META_RESUPPLY   # directly resupply at the nearest capture point

# rewards / penalties
R_CAPTURE_A_C   = efg.R_CAPTURE_A_C
R_CAPTURE_B     = efg.R_CAPTURE_B
R_DESTROY_ENEMY = efg.R_DESTROY_ENEMY   # higher to make killing enemies worth delegating to attack_agent
R_RESUPPLY      = efg.R_RESUPPLY
R_WIN           = efg.R_WIN
P_LOSE          = efg.P_LOSE
P_STEP          = efg.P_STEP

# observation vector size (one per blue peloton)
OBS_SIZE = efg.OBS_SIZE

# Creates a class which as implmented t
class NormandyEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, seed=efg.SEED, render_every=efg.RENDER_EVERY):
        super().__init__()

        self.render_mode   = render_mode
        self.render_every  = render_every   # render once every N episodes (0 = every step)
        self.map_gen = MapGenerator(size=MAP_SIZE, seed=seed)
        # generate the map ONCE and reuse it — the terrain stays the same every episode
        # so that Q-tables can actually converge to a stable navigation policy
        self.fixed_map = self.map_gen.generate_map()


        # commander sends one META-action per blue peloton (who controls this step)
        self.action_space = spaces.MultiDiscrete([4] * NUM_BLUE)

        # each blue peloton has a flat observation vector of 16 values (range 0-9)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0.0, high=9.0, shape=(OBS_SIZE,), dtype=np.float32)
            for _ in range(NUM_BLUE)
        ])
        self.episode = 0
        # one set of sub-agents per blue peloton
        self.attack_agents  = [attack_agent()  for _ in range(NUM_BLUE)]
        self.defense_agents = [defense_agent() for _ in range(NUM_BLUE)]
        self.capture_agents = [capture_agent() for _ in range(NUM_BLUE)]

        # one set of sub-agents per red peloton (same structure as blue)
        self.red_command_agents = [command_agent() for _ in range(NUM_RED)]
        self.red_attack_agents  = [attack_agent()  for _ in range(NUM_RED)]
        self.red_defense_agents = [defense_agent() for _ in range(NUM_RED)]
        self.red_capture_agents = [capture_agent() for _ in range(NUM_RED)]
        self.red_captured = {'A': False, 'B': False, 'C': False}

        # set map now so _find_free_cell works below
        self.map = self.fixed_map 

        # find starting positions ONCE and reuse every episode
        # same terrain + same starts = Q-tables can actually converge
        self.fixed_blue_starts = [self._find_free_cell(0, 6, 18, 24) for _ in range(NUM_BLUE)]
        self.fixed_red_starts  = [self._find_free_cell(18, 24, 0, 6) for _ in range(NUM_RED)]

        # initiates the peloton,their points, the points to captura and their steps
        self.blue_pelotons  = []
        self.red_pelotons   = []
        self.points         = {}
        self.captured       = {'A': False, 'B': False, 'C': False}
        self.step_count     = 0

        self.window          = None
        self.clock           = None
        self.tiger_img       = None   # cached so we don't reload every step
        self.sherman_img     = None
        self.map_bg_surface  = None   # optional background image (mapaNormandia.png)
        self.active_explosions = []   # list of {pos, frames_left} for hit effects

        self.reset()

    # Each peloton consists of this caracteristics when it is created
    def _make_peloton(self, x, y, team, pid, hp=500, num_tanks=5):
        return {
            'id':        pid,
            'team':      team,
            'pos':       [x, y],
            'hp':        hp,
            'num_tanks': num_tanks,
            'ammo':      100,
            'fuel':      500,
        }

    
    # verification of a specific position in passable (transitable is bool)
    def _is_passable(self, x, y):
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return False
        return self.map[y][x]['type'] not in IMPASSABLE

    # trys no find an free cell randomly
    def _find_free_cell(self, x_min, x_max, y_min, y_max):
        for _ in range(500):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            if self._is_passable(x, y):
                return x, y
        # fallback: search the whole map
        for _ in range(1000):
            x = random.randint(0, MAP_SIZE - 1)
            y = random.randint(0, MAP_SIZE - 1)
            if self._is_passable(x, y):
                return x, y
        return 0, 0

    # resets the enviroment with the seed
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.map = self.fixed_map   # same terrain every episode
        self.points = {
            name: {
                'pos':  list(pos),
                'gas':  POINT_SUPPLY_LIMITS['gas'],
                'ammo': POINT_SUPPLY_LIMITS['ammo']
            }
            for name, pos in FIXED_POINTS.items()
        }
        self.captured   = {'A': False, 'B': False, 'C': False}
        self.step_count = 0
        self.active_explosions = []

        # same starting positions every episode (set once in __init__)
        self.blue_pelotons = []
        for i, (x, y) in enumerate(self.fixed_blue_starts):
            self.blue_pelotons.append(self._make_peloton(x, y, 'blue', i, hp=700, num_tanks=7))

        # reds are 3:1 in numbers but weaker per peloton (300hp, 3 tanks)
        self.red_pelotons = []
        for i, (x, y) in enumerate(self.fixed_red_starts):
            self.red_pelotons.append(self._make_peloton(x, y, 'red', i, hp=300, num_tanks=3))

        # resets both red and blue capture positions and capture states
        for ca in self.capture_agents:
            ca.reset_position_history()

        self.red_captured = {'A': False, 'B': False, 'C': False}
        for ca in self.red_capture_agents:
            ca.reset_position_history()

        # returns
        self.obs = self._get_obs()
        return self.obs, self._get_info()

    # defines each step in relation of the reward 
    def step(self, commander_meta_actions):
        self.step_count += 1
        rewards     = [0.0] * NUM_BLUE
        red_rewards = [0.0] * NUM_RED

        # snapshot red HP now so we can detect which reds got hit by blue this step
        red_hp_before_blue = [p['hp'] for p in self.red_pelotons]

        # snapshot blue state before anything happens
        pre = self._snapshot_states()

        # track what each sub-agent chose (needed for Q-table updates later)
        atk_actions_taken = [None] * NUM_BLUE
        def_actions_taken = [None] * NUM_BLUE
        cap_actions_taken = [None] * NUM_BLUE
        hit_confirmed     = [False] * NUM_BLUE

        # if the vector of the snapshots is empty 
        for i, pel in enumerate(self.blue_pelotons):
            if pre[i] is None:
                continue

            ps   = pre[i]
            meta = int(commander_meta_actions[i])

            # all 3 sub-agents CHOOSE their action every step (they always observe)
            atk_a = self.attack_agents[i].choose_action(ps['atk_state'])
            def_a = self.defense_agents[i].choose_action(ps['enemy_nearby'], ps['cover_type'])

            # it gives the value of where the nearest object is
            nearest_obj = self._nearest_uncaptured_point(pel)


            if nearest_obj is not None:
                cap_a = self.capture_agents[i].choose_action(tuple(pel['pos']), tuple(nearest_obj['pos']))
            else:
                cap_a = STAY

            # defines each step what the decision was
            atk_actions_taken[i] = atk_a
            def_actions_taken[i] = def_a
            cap_actions_taken[i] = cap_a

            # ONLY the delegated sub-agent executes its action, depending if he want's to attack, defend or resupply
            if meta == META_ATTACK:
                if atk_a == SHOOT and len(ps['enemies_in_range']) > 0:
                    target = min(ps['enemies_in_range'], key=lambda e: distance(pel['pos'], e['pos']))
                    
                    if distance(pel['pos'], target['pos']) <= ATTACK_RANGE:
                        old_tanks    = target['num_tanks']
                        target_cover = self.map[target['pos'][1]][target['pos'][0]]['cover']
                        dmg = do_attack(pel, target, target_cover, damage_per_tank=TIGER_DAMAGE)

                        if dmg > 0:
                            hit_confirmed[i] = True
                            rewards[i] += dmg * 0.5   # bigger signal per hit
                            self.active_explosions.append({'pos': list(target['pos']), 'frames_left': 4})
                        if old_tanks > 0 and target['num_tanks'] <= 0:
                            rewards[i] += R_DESTROY_ENEMY

            elif meta == META_DEFENSE:
                if def_a == TAKE_COVER:
                    cover_cell = get_best_cover_cell(pel, self.map, MAP_SIZE)
                    if cover_cell is not None:
                        move_cost = self.map[cover_cell[1]][cover_cell[0]].get('penalization', 0) + 1
                        if pel['fuel'] >= move_cost:
                            pel['pos']   = list(cover_cell)
                            pel['fuel'] -= move_cost

            elif meta == META_CAPTURE:
                # capture_agent already chose direction, now we execute it
                dx, dy = 0, 0
                if cap_a == MOVE_UP:
                    dy = -1
                elif cap_a == MOVE_DOWN:
                    dy = 1
                elif cap_a == MOVE_RIGHT:
                    dx = 1
                elif cap_a == MOVE_LEFT:
                    dx = -1
                # STAY: dx=dy=0, no movement

                new_x = pel['pos'][0] + dx
                new_y = pel['pos'][1] + dy
                if (dx != 0 or dy != 0) and self._is_passable(new_x, new_y):
                    move_cost = self.map[new_y][new_x].get('penalization', 0) + 1
                    if pel['fuel'] >= move_cost:
                        pel['pos']   = [new_x, new_y]
                        pel['fuel'] -= move_cost

            elif meta == META_RESUPPLY:
                for point_data in self.points.values():
                    if pel['pos'] == point_data['pos']:
                        if do_resupply(pel, point_data):
                            rewards[i] += R_RESUPPLY
                        break

            # check if standing on a capture point after any movement
            for point_name, point_data in self.points.items():
                if pel['pos'] == point_data['pos'] and not self.captured[point_name]:
                    self.captured[point_name] = True
                    if self.red_captured[point_name]:
                        self.red_captured[point_name] = False  # blue reconquers it from red
                    rewards[i] += R_CAPTURE_B if point_name == 'B' else R_CAPTURE_A_C

        # snapshot hp before red attacks (to detect got_hit for blue later)
        blue_hp_before_red = [p['hp'] for p in self.blue_pelotons]

        # snapshot red state before red acts (used for red sub-agent learning)
        pre_red = self._snapshot_red_states()

        # track what each red sub-agent chose
        red_atk_actions = [None] * NUM_RED
        red_def_actions = [None] * NUM_RED
        red_cap_actions = [None] * NUM_RED
        red_meta_taken  = [None] * NUM_RED
        red_hit_confirmed = [False] * NUM_RED

        # RED AGENT TURN — mirrors the blue loop above
        for i, red_pel in enumerate(self.red_pelotons):
            if pre_red[i] is None:
                continue

            psr  = pre_red[i]
            meta = self.red_command_agents[i].choose_action(psr['obs_vec'])
            red_meta_taken[i] = meta

            atk_a = self.red_attack_agents[i].choose_action(psr['atk_state'])
            def_a = self.red_defense_agents[i].choose_action(psr['enemy_nearby'], psr['cover_type'])

            nearest_obj_r = self._nearest_uncaptured_point_red(red_pel)
            if nearest_obj_r is not None:
                cap_a = self.red_capture_agents[i].choose_action(
                    tuple(red_pel['pos']), tuple(nearest_obj_r['pos'])
                )
            else:
                cap_a = STAY

            red_atk_actions[i] = atk_a
            red_def_actions[i] = def_a
            red_cap_actions[i] = cap_a

            if meta == META_ATTACK:
                blues_in_range = get_enemies_in_range(red_pel, self.blue_pelotons, ATTACK_RANGE)
                if atk_a == SHOOT and len(blues_in_range) > 0:
                    target = min(blues_in_range, key=lambda e: distance(red_pel['pos'], e['pos']))
                    old_tanks    = target['num_tanks']
                    target_cover = self.map[target['pos'][1]][target['pos'][0]]['cover']
                    dmg = do_attack(red_pel, target, target_cover)  # Sherman damage (default)
                    if dmg > 0:
                        red_hit_confirmed[i] = True
                        red_rewards[i] += dmg * 0.5
                        self.active_explosions.append({'pos': list(target['pos']), 'frames_left': 4})
                    if old_tanks > 0 and target['num_tanks'] <= 0:
                        red_rewards[i] += R_DESTROY_ENEMY

            elif meta == META_DEFENSE:
                if def_a == TAKE_COVER:
                    cover_cell = get_best_cover_cell(red_pel, self.map, MAP_SIZE)
                    if cover_cell is not None:
                        move_cost = self.map[cover_cell[1]][cover_cell[0]].get('penalization', 0) + 1
                        if red_pel['fuel'] >= move_cost:
                            red_pel['pos']   = list(cover_cell)
                            red_pel['fuel'] -= move_cost

            elif meta == META_CAPTURE:
                dx, dy = 0, 0
                if cap_a == MOVE_UP:    dy = -1
                elif cap_a == MOVE_DOWN:  dy =  1
                elif cap_a == MOVE_RIGHT: dx =  1
                elif cap_a == MOVE_LEFT:  dx = -1

                new_x = red_pel['pos'][0] + dx
                new_y = red_pel['pos'][1] + dy
                if (dx != 0 or dy != 0) and self._is_passable(new_x, new_y):
                    move_cost = self.map[new_y][new_x].get('penalization', 0) + 1
                    if red_pel['fuel'] >= move_cost:
                        red_pel['pos']   = [new_x, new_y]
                        red_pel['fuel'] -= move_cost

            elif meta == META_RESUPPLY:
                for point_data in self.points.values():
                    if red_pel['pos'] == point_data['pos']:
                        if do_resupply(red_pel, point_data):
                            red_rewards[i] += R_RESUPPLY
                        break

            # check if red is standing on a capture point after movement
            for point_name, point_data in self.points.items():
                if red_pel['pos'] == point_data['pos'] and not self.red_captured[point_name]:
                    self.red_captured[point_name] = True
                    if self.captured[point_name]:
                        self.captured[point_name] = False  # red reconquers it from blue
                    red_rewards[i] += R_CAPTURE_B if point_name == 'B' else R_CAPTURE_A_C

        # BLUE SUB-AGENTS LEARN (after red attacked, so got_hit is accurate)
        for i, pel in enumerate(self.blue_pelotons):
            if pre[i] is None: # Skip destroyed platoons (no learning update)
                continue

            ps      = pre[i]
            got_hit = pel['hp'] < blue_hp_before_red[i]  # Check if platoon took damage after red attack

            # Get new enemies in range and next attack state
            new_in_range   = get_enemies_in_range(pel, self.red_pelotons, ATTACK_RANGE)
            next_atk_state = self.attack_agents[i].get_state(new_in_range)

            # Compute reward based on previous state, action taken, and hit outcome
            atk_reward     = self.attack_agents[i].compute_reward(
                ps['atk_state'], atk_actions_taken[i], hit_confirmed[i]
            )  # Compute reward based on previous state, action taken, and hit outcome
            self.attack_agents[i].update(ps['atk_state'], atk_actions_taken[i], atk_reward, next_atk_state)

            # Compute next enemy proximity
            nearest_now, dist_now = get_nearest_enemy(pel, self.red_pelotons)
            next_enemy_nearby = 1 if (nearest_now is not None and dist_now <= 4) else 0
            next_cover_val    = self.map[pel['pos'][1]][pel['pos'][0]]['cover']
            next_cover_type   = get_cover_type_int(next_cover_val)
            def_reward = self.defense_agents[i].compute_reward(
                ps['enemy_nearby'], ps['cover_type'], def_actions_taken[i], got_hit, next_cover_type
            )
            self.defense_agents[i].update(
                ps['enemy_nearby'], ps['cover_type'],
                def_actions_taken[i], def_reward,
                next_enemy_nearby, next_cover_type
            )

            # Find nearest uncaptured objective
            nearest_obj = self._nearest_uncaptured_point(pel)
            if nearest_obj is not None:
                old_pos_t = tuple(ps['pos'])
                new_pos_t = tuple(pel['pos'])
                obj_pos_t = tuple(nearest_obj['pos'])
                cap_reward = self.capture_agents[i].compute_reward(old_pos_t, new_pos_t, obj_pos_t)
                self.capture_agents[i].update(
                    old_pos_t, obj_pos_t, cap_actions_taken[i], cap_reward, new_pos_t
                )
                self.capture_agents[i].update_position_history(new_pos_t)

        # RED SUB-AGENTS + COMMAND AGENTS LEARN
        red_obs_after = self._get_red_obs()
        for i, red_pel in enumerate(self.red_pelotons):
            if pre_red[i] is None:
                continue

            # Check if red platoon took damage
            psr     = pre_red[i]
            got_hit_red = red_pel['hp'] < red_hp_before_blue[i]

              #  Attack agent (red) 
            new_blues_in_range = get_enemies_in_range(red_pel, self.blue_pelotons, ATTACK_RANGE)
            next_atk_state_r   = self.red_attack_agents[i].get_state(new_blues_in_range)
            atk_reward_r       = self.red_attack_agents[i].compute_reward(
                psr['atk_state'], red_atk_actions[i], red_hit_confirmed[i]
            )
            self.red_attack_agents[i].update(psr['atk_state'], red_atk_actions[i], atk_reward_r, next_atk_state_r)

            #  Defense agent (red) 
            nearest_blue_now, dist_blue_now = get_nearest_enemy(red_pel, self.blue_pelotons)
            next_enemy_nearby_r = 1 if (nearest_blue_now is not None and dist_blue_now <= 4) else 0
            next_cover_val_r    = self.map[red_pel['pos'][1]][red_pel['pos'][0]]['cover']
            next_cover_type_r   = get_cover_type_int(next_cover_val_r)
            def_reward_r = self.red_defense_agents[i].compute_reward(
                psr['enemy_nearby'], psr['cover_type'], red_def_actions[i], got_hit_red, next_cover_type_r
            )
            self.red_defense_agents[i].update(
                psr['enemy_nearby'], psr['cover_type'],
                red_def_actions[i], def_reward_r,
                next_enemy_nearby_r, next_cover_type_r
            )

            nearest_obj_r = self._nearest_uncaptured_point_red(red_pel)
            if nearest_obj_r is not None:
                old_pos_r = tuple(psr['pos'])
                new_pos_r = tuple(red_pel['pos'])
                obj_pos_r = tuple(nearest_obj_r['pos'])
                cap_reward_r = self.red_capture_agents[i].compute_reward(old_pos_r, new_pos_r, obj_pos_r)
                self.red_capture_agents[i].update(
                    old_pos_r, obj_pos_r, red_cap_actions[i], cap_reward_r, new_pos_r
                )
                self.red_capture_agents[i].update_position_history(new_pos_r)

            # command agent update for red
            new_obs_r  = red_obs_after[i]
            cmd_r      = self.red_command_agents[i].compute_reward(
                psr['obs_vec'], new_obs_r, red_meta_taken[i], red_rewards[i]
            )
            self.red_command_agents[i].update(psr['obs_vec'], red_meta_taken[i], cmd_r, new_obs_r)

        # Step penalty (encourage efficiency)
        for i in range(NUM_BLUE):
            rewards[i] += P_STEP
        for i in range(NUM_RED):
            red_rewards[i] += P_STEP

        terminated = False

        # Blue wins by capturing all objectives
        if all(self.captured.values()):
            terminated = True
            for i in range(NUM_BLUE):
                rewards[i] += R_WIN
            for i in range(NUM_RED):
                red_rewards[i] += P_LOSE

        # Red wins by capturing all objectives
        if all(self.red_captured.values()):
            terminated = True
            for i in range(NUM_RED):
                red_rewards[i] += R_WIN
            for i in range(NUM_BLUE):
                rewards[i] += P_LOSE

        # Red wins if all blue units destroyed
        if all_dead(self.blue_pelotons):
            terminated = True
            for i in range(NUM_BLUE):
                rewards[i] += P_LOSE
            for i in range(NUM_RED):
                red_rewards[i] += R_WIN

        # Blue wins if all red units destroyed
        if all_dead(self.red_pelotons):
            terminated = True
            for i in range(NUM_BLUE):
                rewards[i] += R_WIN
            for i in range(NUM_RED):
                red_rewards[i] += P_LOSE

        truncated = False

        # Decay exploration (epsilon) after each episode
        if terminated or truncated:
            for i in range(NUM_BLUE):
                self.attack_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.defense_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.capture_agents[i].decay_epsilon(decay_rate=0.9995, min_epsilon=0.05)
            for i in range(NUM_RED):
                self.red_command_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.red_attack_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.red_defense_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.red_capture_agents[i].decay_epsilon(decay_rate=0.9995, min_epsilon=0.05)

        if self.render_mode == "human":
            self._render()

        self.obs = self._get_obs() # Compute next observation

        return self.obs, rewards, terminated, truncated, self._get_info()
    
    # helpers used inside step()
   
    def _snapshot_states(self):
        pre = []
        for i, pel in enumerate(self.blue_pelotons):
            if pel['num_tanks'] <= 0:
                pre.append(None)
                continue

            enemies_in_range = get_enemies_in_range(pel, self.red_pelotons, ATTACK_RANGE)
            atk_state = self.attack_agents[i].get_state(enemies_in_range)

            nearest_enemy, nearest_dist = get_nearest_enemy(pel, self.red_pelotons)
            enemy_nearby = 1 if (nearest_enemy is not None and nearest_dist <= 4) else 0
            cover_val    = self.map[pel['pos'][1]][pel['pos'][0]]['cover']
            cover_type   = get_cover_type_int(cover_val)

            pre.append({
                'hp':               pel['hp'],
                'pos':              list(pel['pos']),
                'atk_state':        atk_state,
                'enemy_nearby':     enemy_nearby,
                'cover_type':       cover_type,
                'enemies_in_range': enemies_in_range,
            })
        return pre

    
    def _nearest_uncaptured_point(self, pel):
        # Find closest uncaptured objective for blue team
        best      = None
        best_dist = 99999
        for name, pd in self.points.items():
            if not self.captured[name]:
                d = distance(pel['pos'], pd['pos'])
                if d < best_dist:
                    best_dist = d
                    best = pd
        return best

   
    def _nearest_uncaptured_point_red(self, pel):
         # Find closest uncaptured objective for red team
        best      = None
        best_dist = 99999
        for name, pd in self.points.items():
            if not self.red_captured[name]:
                d = distance(pel['pos'], pd['pos'])
                if d < best_dist:
                    best_dist = d
                    best = pd
        return best
    
    # captures and saves the actual states of the red pelotons in an specific moment
    def _snapshot_red_states(self):
        pre = []
        for i, red_pel in enumerate(self.red_pelotons):
            if red_pel['num_tanks'] <= 0:
                pre.append(None)
                continue

            blues_in_range = get_enemies_in_range(red_pel, self.blue_pelotons, ATTACK_RANGE)
            atk_state = self.red_attack_agents[i].get_state(blues_in_range)

            # Compute proximity to nearest blue
            nearest_blue, nearest_dist = get_nearest_enemy(red_pel, self.blue_pelotons)
            enemy_nearby = 1 if (nearest_blue is not None and nearest_dist <= 4) else 0
            cover_val    = self.map[red_pel['pos'][1]][red_pel['pos'][0]]['cover']
            cover_type   = get_cover_type_int(cover_val)

            # Store state information
            pre.append({
            'hp':               red_pel['hp'],                        # current health
            'pos':              list(red_pel['pos']),                 # position
            'atk_state':        atk_state,                            # attack state encoding
            'enemy_nearby':     enemy_nearby,                         # proximity flag
            'cover_type':       cover_type,                           # terrain type
            'enemies_in_range': blues_in_range,                       # attackable enemies
            'obs_vec':          self._build_red_obs_single(red_pel),  # full observation vector
        })

        return pre

    # Used to get the observations for the red agent
    def _build_red_obs_single(self, red_pel):
        nearest_blue, blue_dist = get_nearest_enemy(red_pel, self.blue_pelotons)
        if nearest_blue is None:
            enemy_nearby       = 0 # no enemies alive
            enemy_dist_clamped = 9
        else:
            enemy_nearby       = 1 if blue_dist <= 4 else 0
            enemy_dist_clamped = min(blue_dist, 9)

        # Objective features
        nearest_obj = self._nearest_uncaptured_point_red(red_pel)
        if nearest_obj is not None:

            # Relative position to objective
            obj_dx     = nearest_obj['pos'][0] - red_pel['pos'][0]
            obj_dy     = nearest_obj['pos'][1] - red_pel['pos'][1]
              # Direction encoding (0 = same, 1 = positive, 2 = negative)
            obj_dx_dir = 0 if obj_dx == 0 else (1 if obj_dx > 0 else 2)
            obj_dy_dir = 0 if obj_dy == 0 else (1 if obj_dy > 0 else 2)
            obj_dist   = min(abs(obj_dx) + abs(obj_dy), 9)
        else:
            obj_dx_dir = 0
            obj_dy_dir = 0
            obj_dist   = 0

          # Get cover type at current position
        cover_type = get_cover_type_int(self.map[red_pel['pos'][1]][red_pel['pos'][0]]['cover'])

        return np.array([
            red_pel['hp'] // 100,                # 0–5  HP scaled in hundreds
            red_pel['fuel'] // 20,               # 0–5  fuel level (coarse discretization)
            red_pel['ammo'] // 20,               # 0–5  ammo level (coarse discretization)
            red_pel['num_tanks'],                # 0–5  number of tanks remaining
            cover_type,                          # 0–2  cover type at current position
            enemy_nearby,                        # 0–1  whether an enemy is within 4 cells
            enemy_dist_clamped,                  # 0–9  distance to nearest enemy (clamped)
            1 if self.red_captured['A'] else 0,  # 0–1  objective A captured
            1 if self.red_captured['B'] else 0,  # 0–1  objective B captured
            1 if self.red_captured['C'] else 0,  # 0–1  objective C captured
            obj_dx_dir,                          # 0–2  direction to nearest objective (x-axis)
            obj_dy_dir,                          # 0–2  direction to nearest objective (y-axis)
            obj_dist,                            # 0–9  distance to nearest objective
            red_pel['pos'][0] // 5,              # 0–4  map sector (x coordinate)
            red_pel['pos'][1] // 5,              # 0–4  map sector (y coordinate)
            1 if red_pel['ammo'] < 20 else 0,    # 0–1  low ammo indicator
        ], dtype=np.float32)

     # it returns a list with the observations of the red agent
    def _get_red_obs(self):
        return [
            self._build_red_obs_single(red_pel) if red_pel['num_tanks'] > 0
            else np.zeros(OBS_SIZE, dtype=np.float32)
            for red_pel in self.red_pelotons
        ]

    # it returns a list with all the observations of the peloton
    def _get_obs(self):
        # Build observation vector for each blue platoon
        obs_list = []

        for pel in self.blue_pelotons:
            if pel['num_tanks'] <= 0:  # If platoon is destroyed, return a zero observation
                obs_list.append(np.zeros(OBS_SIZE, dtype=np.float32))
                continue
            
            # Get closest enemy platoon and its distance
            nearest_enemy, enemy_dist = get_nearest_enemy(pel, self.red_pelotons)
            if nearest_enemy is None:
                enemy_nearby       = 0  # No enemies alive → default values
                enemy_dist_clamped = 9
            else:  # Binary flag if enemy is within engagement range (<= 4 cells)
                enemy_nearby       = 1 if enemy_dist <= 4 else 0
                enemy_dist_clamped = min(enemy_dist, 9)

            nearest_obj = self._nearest_uncaptured_point(pel)   # Find nearest uncaptured objective
            if nearest_obj is not None:   # Relative position to objective (grid space)
                obj_dx     = nearest_obj['pos'][0] - pel['pos'][0]
                obj_dy     = nearest_obj['pos'][1] - pel['pos'][1]
                obj_dx_dir = 0 if obj_dx == 0 else (1 if obj_dx > 0 else 2)
                obj_dy_dir = 0 if obj_dy == 0 else (1 if obj_dy > 0 else 2)
                obj_dist   = min(abs(obj_dx) + abs(obj_dy), 9)
            else:   # No objectives remaining
                obj_dx_dir = 0
                obj_dy_dir = 0
                obj_dist   = 0

            cover_type = get_cover_type_int(self.map[pel['pos'][1]][pel['pos'][0]]['cover'])

            obs = np.array([
                pel['hp'] // 100,                # 0-5  hp in hundreds
                pel['fuel'] // 20,               # 0-5  fuel level
                pel['ammo'] // 20,               # 0-5  ammo level
                pel['num_tanks'],                # 0-5  tanks remaining
                cover_type,                      # 0-2  current cell cover
                enemy_nearby,                    # 0-1  enemy within 4 cells
                enemy_dist_clamped,              # 0-9  distance to nearest enemy
                1 if self.captured['A'] else 0,  # 0-1
                1 if self.captured['B'] else 0,  # 0-1
                1 if self.captured['C'] else 0,  # 0-1
                obj_dx_dir,                      # 0-2  direction to nearest objective (x)
                obj_dy_dir,                      # 0-2  direction to nearest objective (y)
                obj_dist,                        # 0-9  distance to nearest objective
                pel['pos'][0] // 5,              # 0-4  map sector x
                pel['pos'][1] // 5,              # 0-4  map sector y
                1 if pel['ammo'] < 20 else 0,    # 0-1  low ammo flag
            ], dtype=np.float32)

            obs_list.append(obs)

        return obs_list

    # get's general info in any moment of the battle
    def _get_info(self):
        return {
            'step':         self.step_count,
            'blue_alive':   sum(1 for p in self.blue_pelotons if p['num_tanks'] > 0),
            'red_alive':    sum(1 for p in self.red_pelotons  if p['num_tanks'] > 0),
            'captured':     self.captured.copy(),
            'red_captured': self.red_captured.copy(),
            'red_eps':      self.red_command_agents[0].epsilon,
        }

    # goes to next episode
    def increase_episode(self):
        self.episode += 1

    # renderers the py game
    def _render(self):
        
        if self.render_mode != "human":
            return

        # skip render unless this episode is a multiple of render_every
        if self.render_every > 0 and self.episode % self.render_every != 0:
            return

        try:
            import pygame
        except ImportError:
            print("pygame not installed, cannot render")
            return

        cell_size = 750 // MAP_SIZE

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((750, 750))
            self.clock  = pygame.time.Clock()
            # try to load map background image (optional)
            try:
                bg = pygame.image.load("resources/mapaNormandia.png").convert()
                self.map_bg_surface = pygame.transform.scale(bg, (750, 750))
            except Exception:
                self.map_bg_surface = None
            # load unit images once — reuse every step instead of re-loading every frame
            try:
                self.tiger_img = pygame.transform.scale(
                    pygame.image.load("resources/tiger.png").convert_alpha(), (cell_size, cell_size))
                self.sherman_img = pygame.transform.scale(
                    pygame.image.load("resources/sherman.png").convert_alpha(), (cell_size, cell_size))
            except Exception:
                self.tiger_img   = None
                self.sherman_img = None

        pygame.event.pump()   # keeps the OS from marking the window as "not responding"
        pygame.display.set_caption(f"Normandy RL — ep {self.episode}  step {self.step_count}")

        # the different colors to diferentiate the map on the screen
        terrain_colors = {
            'OPEN':   (100, 200, 100),
            'BUSH':   (50,  150,  50),
            'FOREST': (20,  100,  20),
            'RUBBLE': (120, 120, 120),
            'WALL':   (80,   80,  80),
            'WATER':  (50,  100, 200),
        }

        # background: image if available, black otherwise
        if self.map_bg_surface:
            self.window.blit(self.map_bg_surface, (0, 0))
        else:
            self.window.fill((0, 0, 0))

        # terrain grid — semi-transparent when background image is present so both are visible
        terrain_overlay = pygame.Surface((750, 750), pygame.SRCALPHA)
        cell_alpha = 110 if self.map_bg_surface else 255
        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                cell  = self.map[row][col]

                # Get terrain color 
                r, g, b = terrain_colors.get(cell['type'], (200, 200, 200))

                # Compute cell rectangle in screen coordinates
                rect  = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                # Fill cell and draw a subtle border
                pygame.draw.rect(terrain_overlay, (r, g, b, cell_alpha), rect)
                pygame.draw.rect(terrain_overlay, (0, 0, 0, 60), rect, 1)
        self.window.blit(terrain_overlay, (0, 0))

        # draw capture points (yellow = free, cyan = captured)
        font_small = pygame.font.Font(None, 18)
        for name, p in self.points.items():
            cx = p['pos'][0] * cell_size + cell_size // 2
            cy = p['pos'][1] * cell_size + cell_size // 2
            color = (0, 200, 255) if self.captured[name] else (255, 215, 0)
            pygame.draw.circle(self.window, color, (cx, cy), cell_size // 3)
            lbl = font_small.render(name, True, (0, 0, 0))
            self.window.blit(lbl, (cx - 4, cy - 6))

        for pel in self.blue_pelotons:
            if pel['num_tanks'] > 0:
                xp = pel['pos'][0] * cell_size
                yp = pel['pos'][1] * cell_size
                if self.tiger_img:
                    self.window.blit(self.tiger_img, (xp, yp))
                else:
                    pygame.draw.rect(self.window, (30, 80, 220), pygame.Rect(xp, yp, cell_size, cell_size))
                txt = font_small.render(str(pel['num_tanks']), True, (255, 255, 255))
                self.window.blit(txt, (xp + 2, yp + 2))

        for pel in self.red_pelotons:
            if pel['num_tanks'] > 0:
                xp = pel['pos'][0] * cell_size
                yp = pel['pos'][1] * cell_size
                if self.sherman_img:
                    self.window.blit(self.sherman_img, (xp, yp))
                else:
                    pygame.draw.rect(self.window, (200, 30, 30), pygame.Rect(xp, yp, cell_size, cell_size))

        # episode + step HUD (top-left, dark background so it's always readable)
        font_hud  = pygame.font.Font(None, 28)
        hud_text  = f"ep {self.episode}   step {self.step_count}"
        hud_surf  = font_hud.render(hud_text, True, (255, 255, 255))
        hud_rect  = pygame.Rect(4, 4, hud_surf.get_width() + 12, 28)
        pygame.draw.rect(self.window, (0, 0, 0), hud_rect)
        self.window.blit(hud_surf, (10, 8))

        # draw explosion effects on hit tanks
        still_alive = []
        for exp in self.active_explosions:
            # Convert grid position to pixel center
            ex = exp['pos'][0] * cell_size + cell_size // 2
            ey = exp['pos'][1] * cell_size + cell_size // 2

            # Fade-out effect based on remaining frames
            alpha = int(220 * exp['frames_left'] / 4)
            radius = cell_size // 2 + 4
            exp_surf = pygame.Surface((cell_size * 3, cell_size * 3), pygame.SRCALPHA)
            center = (cell_size + cell_size // 2, cell_size + cell_size // 2)

            # Draw explosion layers (outer → inner)
            pygame.draw.circle(exp_surf, (255, 200, 0, alpha),            center, radius)
            pygame.draw.circle(exp_surf, (255, 80,  0, min(255, alpha + 40)), center, radius // 2)
            pygame.draw.circle(exp_surf, (255, 255, 200, min(255, alpha + 80)), center, radius // 4)
            self.window.blit(exp_surf, (ex - cell_size - cell_size // 2, ey - cell_size - cell_size // 2))
            
            exp['frames_left'] -= 1 # decrease lifetime

            # keep explosion if still active
            if exp['frames_left'] > 0:
                still_alive.append(exp)
        self.active_explosions = still_alive

        # final render
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    # for the closing of the screen
    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None


# implements the wrappers used in wrappers.py to the enviroment
def make_env(render_mode=None, max_steps=500, fog_of_war=True, action_mask=True, render_every=efg.RENDER_EVERY):
    from env.wrappers import FogOfWarWrapper, ActionMaskWrapper, EpisodeStatsWrapper # implementing libraries
    
    env = NormandyEnv(render_mode=render_mode, render_every=render_every)
    if fog_of_war:
        env = FogOfWarWrapper(env)
    if action_mask:
        env = ActionMaskWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = EpisodeStatsWrapper(env)
    return env # Creating all the wrappers in relation with wrappers

    
"""
normandy_env.py
Entorno Gymnasium que implementa toda la lógica de la práctica.
Controla 4 pelotones azules, enemigos rojos hardcodeados, puntos, recompensas.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from env.map_generator import MapGenerator, FIXED_POINTS, POINT_SUPPLY_LIMITS, IMPASSABLE
from env.combat_logic import (
    get_enemies_in_range, get_nearest_enemy, do_attack,
    get_best_cover_cell, get_cover_type_int, do_resupply,
    all_dead, ATTACK_RANGE
)
from env.units import distance

from agents.attack_agent  import attack_agent,  SHOOT
from agents.defense_agent import defense_agent, TAKE_COVER
from agents.capture_agent import capture_agent, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY

NUM_BLUE = 4
NUM_RED  = 12
MAP_SIZE = 25
MAX_STEPS = 500

# commander meta-actions: the commander decides WHICH sub-agent takes action on the peloton this step
# movement direction is never chosen by the commander directly — the capture_agent handles that
META_CAPTURE  = 0   # capture_agent decides where to move
META_ATTACK   = 1   # attack_agent decides whether to shoot
META_DEFENSE  = 2   # defense_agent decides whether to seek cover
META_RESUPPLY = 3   # directly resupply at the nearest capture point

# rewards / penalties
R_CAPTURE_A_C   = 100
R_CAPTURE_B     = 200
R_DESTROY_ENEMY = 200   # higher to make killing enemies worth delegating to attack_agent
R_RESUPPLY      = 10
R_WIN           = 1000
P_LOSE          = -500
P_STEP          = -0.1

# observation vector size (one per blue peloton)
OBS_SIZE = 16
# ============================================================================
# CLASE PRINCIPAL DEL ENTORNO
# ============================================================================

class NormandyEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, seed=42):
        super().__init__()

        self.render_mode = render_mode
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

        # one set of sub-agents per blue peloton
        self.attack_agents  = [attack_agent()  for _ in range(NUM_BLUE)]
        self.defense_agents = [defense_agent() for _ in range(NUM_BLUE)]
        self.capture_agents = [capture_agent() for _ in range(NUM_BLUE)]

        # set map now so _find_free_cell works below
        self.map = self.fixed_map

        # find starting positions ONCE and reuse every episode
        # same terrain + same starts = Q-tables can actually converge
        self.fixed_blue_starts = [self._find_free_cell(0, 6, 18, 24) for _ in range(NUM_BLUE)]
        self.fixed_red_starts  = [self._find_free_cell(18, 24, 0, 6) for _ in range(NUM_RED)]

        self.blue_pelotons  = []
        self.red_pelotons   = []
        self.points         = {}
        self.captured       = {'A': False, 'B': False, 'C': False}
        self.step_count     = 0

        self.window = None
        self.clock  = None

        self.reset()

    def _make_peloton(self, x, y, team, pid, hp=500, num_tanks=5):
        return {
            'id':        pid,
            'team':      team,
            'pos':       [x, y],
            'hp':        hp,
            'num_tanks': num_tanks,
            'ammo':      100,
            'fuel':      100,
        }

    def _is_passable(self, x, y):
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return False
        return self.map[y][x]['type'] not in IMPASSABLE

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

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno.
        """
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

        # same starting positions every episode (set once in __init__)
        self.blue_pelotons = []
        for i, (x, y) in enumerate(self.fixed_blue_starts):
            self.blue_pelotons.append(self._make_peloton(x, y, 'blue', i))

        # reds are 3:1 in numbers but weaker per peloton (300hp, 3 tanks)
        self.red_pelotons = []
        for i, (x, y) in enumerate(self.fixed_red_starts):
            self.red_pelotons.append(self._make_peloton(x, y, 'red', i, hp=300, num_tanks=3))

        for ca in self.capture_agents:
            ca.reset_position_history()

        return self._get_obs(), self._get_info()

    def step(self, commander_meta_actions):
        self.step_count += 1
        rewards = [0.0] * NUM_BLUE

        # snapshot state for each agent before anything happens
        pre = self._snapshot_states()

        # track what each sub-agent chose (needed for Q-table updates later)
        atk_actions_taken = [None] * NUM_BLUE
        def_actions_taken = [None] * NUM_BLUE
        cap_actions_taken = [None] * NUM_BLUE
        hit_confirmed     = [False] * NUM_BLUE

        for i, pel in enumerate(self.blue_pelotons):
            if pre[i] is None:
                continue

            ps   = pre[i]
            meta = int(commander_meta_actions[i])

            # all 3 sub-agents CHOOSE their action every step (they always observe)
            atk_a = self.attack_agents[i].choose_action(ps['atk_state'])
            def_a = self.defense_agents[i].choose_action(ps['enemy_nearby'], ps['cover_type'])

            nearest_obj = self._nearest_uncaptured_point(pel)
            if nearest_obj is not None:
                cap_a = self.capture_agents[i].choose_action(tuple(pel['pos']), tuple(nearest_obj['pos']))
            else:
                cap_a = STAY

            atk_actions_taken[i] = atk_a
            def_actions_taken[i] = def_a
            cap_actions_taken[i] = cap_a

            # ONLY the delegated sub-agent executes its action
            if meta == META_ATTACK:
                if atk_a == SHOOT and len(ps['enemies_in_range']) > 0:
                    target = min(ps['enemies_in_range'],
                                 key=lambda e: distance(pel['pos'], e['pos']))
                    if distance(pel['pos'], target['pos']) <= ATTACK_RANGE:
                        old_tanks    = target['num_tanks']
                        target_cover = self.map[target['pos'][1]][target['pos'][0]]['cover']
                        dmg = do_attack(pel, target, target_cover)
                        if dmg > 0:
                            hit_confirmed[i] = True
                            rewards[i] += dmg * 0.5   # bigger signal per hit
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
                    rewards[i] += R_CAPTURE_B if point_name == 'B' else R_CAPTURE_A_C

        # snapshot hp before red attacks (to detect got_hit later)
        blue_hp_before_red = [p['hp'] for p in self.blue_pelotons]

        self._red_turn()

        # all sub-agents LEARN from this step — regardless of who was in control
        for i, pel in enumerate(self.blue_pelotons):
            if pre[i] is None:
                continue

            ps      = pre[i]
            got_hit = pel['hp'] < blue_hp_before_red[i]

            # attack agent update
            new_in_range   = get_enemies_in_range(pel, self.red_pelotons, ATTACK_RANGE)
            next_atk_state = self.attack_agents[i].get_state(new_in_range)
            atk_reward     = self.attack_agents[i].compute_reward(
                ps['atk_state'], atk_actions_taken[i], hit_confirmed[i]
            )
            self.attack_agents[i].update(ps['atk_state'], atk_actions_taken[i], atk_reward, next_atk_state)

            # defense agent update
            nearest_now, dist_now = get_nearest_enemy(pel, self.red_pelotons)
            next_enemy_nearby = 1 if (nearest_now is not None and dist_now <= 4) else 0
            next_cover_val    = self.map[pel['pos'][1]][pel['pos'][0]]['cover']
            next_cover_type   = get_cover_type_int(next_cover_val)
            def_reward = self.defense_agents[i].compute_reward(
                ps['enemy_nearby'], ps['cover_type'], def_actions_taken[i], got_hit
            )
            self.defense_agents[i].update(
                ps['enemy_nearby'], ps['cover_type'],
                def_actions_taken[i], def_reward,
                next_enemy_nearby, next_cover_type
            )

            # capture agent update — uses its chosen action and the actual position outcome
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

        for i in range(NUM_BLUE):
            rewards[i] += P_STEP

        terminated = False
        if all(self.captured.values()):
            terminated = True
            for i in range(NUM_BLUE):
                rewards[i] += R_WIN

        if all_dead(self.blue_pelotons):
            terminated = True
            for i in range(NUM_BLUE):
                rewards[i] += P_LOSE

        truncated = self.step_count >= MAX_STEPS

        if terminated or truncated:
            for i in range(NUM_BLUE):
                # slower decay so agents keep exploring for more episodes
                self.attack_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.defense_agents[i].decay_epsilon(decay_rate=0.999,  min_epsilon=0.05)
                self.capture_agents[i].decay_epsilon(decay_rate=0.9995, min_epsilon=0.05)

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), rewards, terminated, truncated, self._get_info()

    # -------------------------------------------------------------------------
    # helpers used inside step()
    # -------------------------------------------------------------------------

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
        best      = None
        best_dist = 99999
        for name, pd in self.points.items():
            if not self.captured[name]:
                d = distance(pel['pos'], pd['pos'])
                if d < best_dist:
                    best_dist = d
                    best = pd
        return best

    def _red_turn(self):
        """
        Lógica hardcodeada para los rojos:
        - Si hay un azul a distancia <= 5, atacan.
        - Si no, se mueven hacia el azul más cercano (movimiento simple).
        """
        for red in self.red_pelotons:
            if red['num_tanks'] <= 0:
                continue
            # Encontrar azul más cercano
            nearest_blue = None
            min_dist     = 99999
            for blue in self.blue_pelotons:
                if blue['num_tanks'] <= 0:
                    continue
                d = distance(red['pos'], blue['pos'])
                if d < min_dist:
                    min_dist     = d
                    nearest_blue = blue

            if nearest_blue is None:
                continue

            if min_dist <= ATTACK_RANGE:
                target_cover = self.map[nearest_blue['pos'][1]][nearest_blue['pos'][0]]['cover']
                do_attack(red, nearest_blue, target_cover)
            else:
                # move one step toward nearest blue
                dx = nearest_blue['pos'][0] - red['pos'][0]
                dy = nearest_blue['pos'][1] - red['pos'][1]

                if abs(dx) >= abs(dy):
                    step_x = 1 if dx > 0 else -1
                    new_x, new_y = red['pos'][0] + step_x, red['pos'][1]
                else:
                    step_y = 1 if dy > 0 else -1
                    new_x, new_y = red['pos'][0], red['pos'][1] + step_y

                if self._is_passable(new_x, new_y) and red['fuel'] > 0:
                    red['pos']   = [new_x, new_y]
                    red['fuel'] -= 1

    # -------------------------------------------------------------------------
    # observation, info, render
    # -------------------------------------------------------------------------

    def _get_obs(self):
        obs_list = []
        for pel in self.blue_pelotons:
            if pel['num_tanks'] <= 0:
                obs_list.append(np.zeros(OBS_SIZE, dtype=np.float32))
                continue

            nearest_enemy, enemy_dist = get_nearest_enemy(pel, self.red_pelotons)
            if nearest_enemy is None:
                enemy_nearby       = 0
                enemy_dist_clamped = 9
            else:
                enemy_nearby       = 1 if enemy_dist <= 4 else 0
                enemy_dist_clamped = min(enemy_dist, 9)

            nearest_obj = self._nearest_uncaptured_point(pel)
            if nearest_obj is not None:
                obj_dx     = nearest_obj['pos'][0] - pel['pos'][0]
                obj_dy     = nearest_obj['pos'][1] - pel['pos'][1]
                obj_dx_dir = 0 if obj_dx == 0 else (1 if obj_dx > 0 else 2)
                obj_dy_dir = 0 if obj_dy == 0 else (1 if obj_dy > 0 else 2)
                obj_dist   = min(abs(obj_dx) + abs(obj_dy), 9)
            else:
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

        return tuple(obs_list)

    def _get_info(self):
        """
        Información adicional para depuración.
        """
        return {
            'step': self.step_count,
            'blue_alive': sum(1 for p in self.blue_pelotons if p['num_tanks'] > 0),
            'red_alive': sum(1 for p in self.red_pelotons if p['num_tanks'] > 0),
            'captured': self.captured.copy()
        }
    
    def _render(self):
        """
        Renderizado simple con pygame (opcional).
        """
        if self.render_mode != "human":
            return
        try:
            import pygame
        except ImportError:
            print("pygame not installed, cannot render")
            return
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((750, 750))
            pygame.display.set_caption("Normandy RL")
            self.clock = pygame.time.Clock()

        cell_size = 750 // MAP_SIZE
        self.window.fill((0, 0, 0))

        terrain_colors = {
            'OPEN':   (100, 200, 100),
            'BUSH':   (50,  150,  50),
            'FOREST': (20,  100,  20),
            'RUBBLE': (120, 120, 120),
            'WALL':   (80,   80,  80),
            'WATER':  (50,  100, 200),
        }

        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                cell  = self.map[row][col]
                color = terrain_colors.get(cell['type'], (200, 200, 200))
                rect  = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0,0,0), rect, 1)
        
        # Dibujar puntos A,B,C
        for name, p in self.points.items():
            if not self.captured[name]:
                cx = p['pos'][0] * cell_size + cell_size // 2
                cy = p['pos'][1] * cell_size + cell_size // 2
                pygame.draw.circle(self.window, (255, 255, 0), (cx, cy), cell_size // 3)

        font = pygame.font.Font(None, 18)
        for pel in self.blue_pelotons:
            if pel['num_tanks'] > 0:
                rect = pygame.Rect(pel['pos'][0] * cell_size, pel['pos'][1] * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, (30, 80, 220), rect)
                txt = font.render(str(pel['num_tanks']), True, (255, 255, 255))
                self.window.blit(txt, (pel['pos'][0] * cell_size + 2, pel['pos'][1] * cell_size + 2))

        for pel in self.red_pelotons:
            if pel['num_tanks'] > 0:
                rect = pygame.Rect(pel['pos'][0] * cell_size, pel['pos'][1] * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, (200, 30, 30), rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None
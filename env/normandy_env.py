"""
normandy_env.py
Entorno Gymnasium que implementa toda la lógica de la práctica.
Controla 4 pelotones azules, enemigos rojos hardcodeados, puntos, recompensas.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from env.map_generator import MapGenerator, FIXED_POINTS, POINT_SUPPLY_LIMITS
"""
#from env.combat_logic import CombatLogic


# Importamos nuestros módulos
from combat_logic import (move_peloton, apply_combat, can_attack, resupply, 
                          take_cover, update_peloton_state, create_peloton,
                          GAS_PER_MOVE, AMMO_PER_ATTACK, MAX_GAS, MAX_AMMO,
                          TANKS_PER_PELOTON, HP_PER_TANK, PELOTON_MAX_HP)
from map_generator import MapGenerator, TERRAIN_TYPES, MAP_SIZE

# ============================================================================
# CONSTANTES DEL ENTORNO
# ============================================================================

"""

NUM_BLUE_PELOTONS = 4
NUM_RED_PELOTONS = 12   # Ratio 3:1 (12 reds vs 4 blues)
ATTACK_RANGE     = 5            # Maximum cells to be able to attack
OBSERVATION_RADIUS = 5          # Each peloton sees an 11x11 area around itself
# [CUSTOMIZATION] Increase OBSERVATION_RADIUS for more visibility (harder to learn)
# Decrease it for more fog of war (more realistic but harder)


# Possible Actions
ACTION_MOVE_NORTH = 0
ACTION_MOVE_SOUTH = 1
ACTION_MOVE_EAST  = 2
ACTION_MOVE_WEST  = 3
ACTION_ATTACK     = 4
ACTION_TAKE_COVER = 5
ACTION_RESUPPLY   = 6
ACTION_HOLD       = 7

# Rewards
REWARD_CAPTURE_POINT = 100      # Base para capturar A o C
REWARD_CAPTURE_B     = 200      # B es más valioso
REWARD_DAMAGE_ENEMY  = 2        # Por cada punto de daño al enemigo
PENALTY_DAMAGE_SELF  = -1       # Por cada punto de daño recibido
REWARD_SUPPLY        = 10       # Por recoger suministros
PENALTY_TANK_LOST    = -50      # Por perder un tanque
REWARD_WIN           = 1000     # Por capturar todos los puntos
PENALTY_LOSE         = -500     # Por ser aniquilado
STEP_PENALTY         = -0.1     # Penalización por paso para fomentar eficiencia

IMPASSABLE = {"WATER", "WALL"}
OBS_SIZE = 16
# ============================================================================
# CLASE PRINCIPAL DEL ENTORNO
# ============================================================================

class NormandyEnv(gym.Env):
    """
    Entorno multi-agente (4 agentes azules) con Gymnasium.
    El método step recibe una lista de 4 acciones (una por pelotón).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, seed=42):
        super().__init__()

        self.size = MAP_SIZE
        self.render_mode = render_mode
        self.seed = seed 
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generador de mapas
        self.map_gen = MapGenerator(seed=seed)
        #self.combat = CombatLogic()
        
        # Espacios de acción y observación
        # Cada agente tiene 8 acciones discretas
        self.action_space = spaces.MultiDiscrete([8] * NUM_BLUE_PELOTONS)
        
        # Observación: para simplificar, devolvemos una lista de 4 observaciones (una por agente)
        # Cada observación es un diccionario con 'grid' (ventana local) y 'status' (recursos)
        # Gymnasium espera un solo espacio, pero podemos usar spaces.Dict para multi-agente
        # En la práctica, haremos que el step devuelva una lista de observaciones.
        # Para cumplir con la API de Gymnasium, definimos un espacio que contiene 4 observaciones.
        
        # Its an observation space wich is a tuple of NUM_BLUE_PELOTONS con
        #El peloton puede no estar visible con -1 y 1000 será con full HP, fuel y ammo.
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=9, shape=(OBS_SIZE,), dtype=np.float32)
            for _ in range(NUM_BLUE_PELOTONS)
        ])
        """
        CELL_ENEMY = 6
        CELL_ALLY  = 7
        CELL_POINT = 8
        CELL_SELF  = 9 
        """ 
        
        # Estado interno
        self.map = None
        self.points = None          # Diccionario con posiciones y suministros de A,B,C
        self.blue_pelotons = []     # Lista de dicts de pelotones azules
        self.red_pelotons = []      # Lista de dicts de pelotones rojos
        self.captured_points = {'A': False, 'B': False, 'C': False}
        self.step_count = 0
        self.max_steps = 500
        #self.done = False esto servirá para algo?
        
        # Para renderizado (opcional)
        self.window = None
        self.clock = None
        

        # Inicializar
        self.reset()
    
    # You create a peloton
    def _make_peloton(self, x, y, team):
        """Creates a peloton dict at position (x, y)."""
        return {
            "pos":       [x, y],
            "hp":        500,
            "num_tanks": 5,
            "ammo":      100,
            "fuel":      100,
            "cover":     0.0,
            "team":      team,
            "in_cover":  False
        }

    def _is_passable(self, x, y):
        """Returns True if (x, y) is within bounds and walkable terrain."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self.map[y][x]["type"] not in IMPASSABLE
    
    def _find_free_cell(self, x_range, y_range):
        """Returns a random passable (x, y) within the given ranges."""
        while True:
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            if self._is_passable(x, y):
                return x, y
            
    def _manhattan(self, a, b):
        """Manhattan distance between two pelotons."""
        return abs(a["pos"][0] - b["pos"][0]) + abs(a["pos"][1] - b["pos"][1])

    def _clamp_pos(self, peloton):
        """Keeps peloton inside map bounds."""
        peloton["pos"][0] = max(0, min(self.size - 1, peloton["pos"][0]))
        peloton["pos"][1] = max(0, min(self.size - 1, peloton["pos"][1]))


    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 1. Generar nuevo mapa
        self.map    = self.map_gen.generate_map()
        self.points = {
            name: {
                "pos":  list(pos),
                "gas":  POINT_SUPPLY_LIMITS["gas"],
                "ammo": POINT_SUPPLY_LIMITS["ammo"]
            }
            for name, pos in self.map_gen.get_points().items()
        }
        
        # 2. Colocar pelotones azules en posiciones aleatorias libres (esquinas suroeste)
        self.blue_pelotons = []
        start_positions = [(2,2), (2,5), (5,2), (5,5)]  # Fijas para simplificar, podrían ser aleatorias
        for i, (x, y) in enumerate(start_positions):
            # Asegurar que la celda es transitable
            while not self.map_gen.is_passable(self.grid, x, y):
                x = random.randint(0, MAP_SIZE-1)
                y = random.randint(0, MAP_SIZE-1)
            pel = create_peloton(x, y, team='blue')
            pel['id'] = i
            self.blue_pelotons.append(pel)
        
        # 3. Colocar pelotones rojos (hardcodeados, se mueven con reglas simples)
        self.red_pelotons = []
        # Generar en posiciones aleatorias del lado noreste
        for i in range(NUM_RED_PELOTONS):
            x = random.randint(MAP_SIZE-10, MAP_SIZE-1)
            y = random.randint(0, MAP_SIZE-1)
            while not self.map_gen.is_passable(self.grid, x, y):
                x = random.randint(MAP_SIZE-10, MAP_SIZE-1)
                y = random.randint(0, MAP_SIZE-1)
            red = create_peloton(x, y, team='red')
            red['id'] = i
            self.red_pelotons.append(red)
        
        # 4. Reiniciar estado de captura
        self.captured_points = {'A': False, 'B': False, 'C': False}
        self.step_count = 0
        self.done = False
        
        # 5. Obtener observación inicial
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions):
        """
        Ejecuta un paso del entorno.
        actions: lista de 4 enteros (0..7) para cada pelotón azul.
        Retorna: (obs, rewards, terminated, truncated, info)
        """
        self.step_count += 1
        rewards = [0.0] * NUM_BLUE_PELOTONS  # Recompensa individual por agente
        
        # 1. Aplicar acciones de los pelotones azules (en orden)
        for idx, action in enumerate(actions):
            peloton = self.blue_pelotons[idx]
            reward = self._apply_action(peloton, action, idx)
            rewards[idx] += reward
        
        # 2. Movimiento y ataques de los enemigos rojos (hardcodeado)
        self._red_turn()
        
        # 3. Verificar captura de puntos (si un pelotón azul está sobre un punto)
        for pel in self.blue_pelotons:
            for point_name, point_data in self.points.items():
                if (pel['x'] == point_data['x'] and pel['y'] == point_data['y'] 
                    and not self.captured_points[point_name]):
                    self.captured_points[point_name] = True
                    # Recompensa por captura
                    if point_name == 'B':
                        reward_val = REWARD_CAPTURE_B
                    else:
                        reward_val = REWARD_CAPTURE_POINT
                    # Asignar recompensa al agente que capturó
                    idx = pel['id']
                    rewards[idx] += reward_val
        
        # 4. Aplicar penalización por pérdida de tanques (detectar cambio en num_tanks)
        for pel in self.blue_pelotons:
            # Se podría guardar el número anterior en un atributo, pero simplificamos
            pass  # En una implementación completa, se compararía con el estado previo.
        
        # 5. Recompensa negativa por paso (para fomentar rapidez)
        for i in range(NUM_BLUE_PELOTONS):
            rewards[i] += STEP_PENALTY
        
        # 6. Comprobar terminación
        terminated = False
        # Victoria: capturar los tres puntos
        if all(self.captured_points.values()):
            terminated = True
            for i in range(NUM_BLUE_PELOTONS):
                rewards[i] += REWARD_WIN
        # Derrota: todos los pelotones azules destruidos
        if all(p['num_tanks'] == 0 for p in self.blue_pelotons):
            terminated = True
            for i in range(NUM_BLUE_PELOTONS):
                rewards[i] += PENALTY_LOSE
        # Límite de pasos
        truncated = self.step_count >= self.max_steps
        
        # 7. Obtener observaciones e info
        obs = self._get_obs()
        info = self._get_info()
        self.done = terminated or truncated
        
        # 8. Renderizar si es necesario
        if self.render_mode == "human":
            self._render()
        
        # Gymnasium espera una única recompensa (suma) si es un solo agente.
        # Como tenemos multi-agente, devolvemos una lista. Esto no es estándar, pero
        # para adaptarnos a la práctica, retornamos (obs, rewards, terminated, truncated, info)
        # donde rewards es una lista.
        return obs, rewards, terminated, truncated, info
    
    def _apply_action(self, peloton, action, agent_id):
        """
        Aplica una acción a un pelotón y devuelve la recompensa inmediata.
        """
        reward = 0
        
        if action == ACTION_MOVE_NORTH:
            move_peloton(peloton, 0, self.grid)
        elif action == ACTION_MOVE_SOUTH:
            move_peloton(peloton, 1, self.grid)
        elif action == ACTION_MOVE_EAST:
            move_peloton(peloton, 2, self.grid)
        elif action == ACTION_MOVE_WEST:
            move_peloton(peloton, 3, self.grid)
        elif action == ACTION_ATTACK:
            # Atacar al enemigo más cercano en rango
            nearest_enemy = self._get_nearest_enemy(peloton)
            if nearest_enemy:
                distance = abs(peloton['x'] - nearest_enemy['x']) + abs(peloton['y'] - nearest_enemy['y'])
                cover = self.grid[nearest_enemy['y']][nearest_enemy['x']]['cover']
                damage = apply_combat(peloton, nearest_enemy, distance, cover)
                if damage > 0:
                    reward += damage * REWARD_DAMAGE_ENEMY
                    # Si el enemigo muere, recompensa extra?
                    if nearest_enemy['num_tanks'] == 0:
                        reward += 100  # Bonus por destruir un pelotón enemigo
                # Actualizar estado del atacante (munición ya se gastó en apply_combat)
                update_peloton_state(peloton)
        elif action == ACTION_TAKE_COVER:
            success = take_cover(peloton, self.grid)
            if not success:
                reward -= 1  # Pequeña penalización si no puede cubrirse
        elif action == ACTION_RESUPPLY:
            # Buscar si está en un punto no capturado (o incluso capturado pero con suministros)
            for point_name, point_data in self.points.items():
                if peloton['x'] == point_data['x'] and peloton['y'] == point_data['y']:
                    if point_data['gas'] > 0 or point_data['ammo'] > 0:
                        resupply(peloton, point_data)
                        reward += REWARD_SUPPLY
                    break
        elif action == ACTION_HOLD:
            # No hacer nada, solo esperar
            pass
        
        # Después de cualquier acción que no sea ataque, también se actualiza el estado
        # (por si el movimiento provocó caída de tanques? No, solo combate cambia HP)
        # Pero si el pelotón se queda sin gas, etc., no afecta HP.
        return reward
    
    def _get_nearest_enemy(self, peloton):
        """
        Devuelve el pelotón enemigo más cercano (distancia Manhattan).
        """
        enemies = self.red_pelotons
        min_dist = float('inf')
        nearest = None
        for enemy in enemies:
            if enemy['num_tanks'] <= 0:
                continue
            dist = abs(peloton['x'] - enemy['x']) + abs(peloton['y'] - enemy['y'])
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        return nearest
    
    def _red_turn(self):
        """
        Lógica hardcodeada para los rojos:
        - Si hay un azul a distancia <= 5, atacan.
        - Si no, se mueven hacia el azul más cercano (movimiento simple).
        """
        for red in self.red_pelotons:
            if red['num_tanks'] == 0:
                continue
            # Encontrar azul más cercano
            nearest_blue = None
            min_dist = float('inf')
            for blue in self.blue_pelotons:
                if blue['num_tanks'] == 0:
                    continue
                dist = abs(red['x'] - blue['x']) + abs(red['y'] - blue['y'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_blue = blue
            if nearest_blue is None:
                continue
            
            # Atacar si está a rango
            if min_dist <= 5 and can_attack(red, nearest_blue, max_range=5):
                cover = self.grid[nearest_blue['y']][nearest_blue['x']]['cover']
                damage = apply_combat(red, nearest_blue, min_dist, cover)
                # Actualizar estado del azul (posible pérdida de tanques)
                update_peloton_state(nearest_blue)
            else:
                # Moverse hacia el azul (dirección simple)
                dx = nearest_blue['x'] - red['x']
                dy = nearest_blue['y'] - red['y']
                if abs(dx) > abs(dy):
                    if dx > 0:
                        move_peloton(red, ACTION_MOVE_EAST, self.grid)
                    else:
                        move_peloton(red, ACTION_MOVE_WEST, self.grid)
                else:
                    if dy > 0:
                        move_peloton(red, ACTION_MOVE_SOUTH, self.grid)
                    else:
                        move_peloton(red, ACTION_MOVE_NORTH, self.grid)
    
    def _get_obs(self):
        """
        Construye la observación para cada agente azul.
        Cada observación es un diccionario con:
          - 'grid': ventana local de 11x11x5 canales (codificación one-hot del terreno, más cobertura)
          - 'status': vector normalizado [hp_ratio, gas_ratio, ammo_ratio, num_tanks_ratio, point_A_captured, point_B_captured, point_C_captured]
        """
        obs_list = []
        for pel in self.blue_pelotons:
            # Ventana centrada en el pelotón
            px, py = pel['x'], pel['y']
            grid_window = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
            # Recorrer ventana de radio OBSERVATION_RADIUS
            for dy in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS+1):
                for dx in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS+1):
                    wx = px + dx
                    wy = py + dy
                    if 0 <= wx < MAP_SIZE and 0 <= wy < MAP_SIZE:
                        cell = self.grid[wy][wx]
                        # Canal 0: tipo de terreno codificado como entero (0..5)
                        # Para simplificar, usamos un índice numérico
                        terrain_idx = list(TERRAIN_TYPES.keys()).index(cell['type'])
                        grid_window[dy+OBSERVATION_RADIUS, dx+OBSERVATION_RADIUS, 0] = terrain_idx / 5.0
                        # Canal 1: cobertura
                        grid_window[dy+OBSERVATION_RADIUS, dx+OBSERVATION_RADIUS, 1] = cell['cover']
                        # Canal 2: si hay enemigo rojo
                        enemy_present = 0
                        for red in self.red_pelotons:
                            if red['x'] == wx and red['y'] == wy and red['num_tanks'] > 0:
                                enemy_present = 1
                                break
                        grid_window[dy+OBSERVATION_RADIUS, dx+OBSERVATION_RADIUS, 2] = enemy_present
                        # Canal 3: si hay punto de interés (A/B/C) no capturado
                        point_present = 0
                        for pname, pdata in self.points.items():
                            if pdata['x'] == wx and pdata['y'] == wy and not self.captured_points[pname]:
                                point_present = 1
                                break
                        grid_window[dy+OBSERVATION_RADIUS, dx+OBSERVATION_RADIUS, 3] = point_present
                        # Canal 4: cobertura especial (podría ser otro factor)
                        grid_window[dy+OBSERVATION_RADIUS, dx+OBSERVATION_RADIUS, 4] = 0.0  # reservado
                    else:
                        # Fuera del mapa: rellenar con 0
                        pass
            
            # Estado interno del pelotón (normalizado)
            hp_ratio = pel['hp'] / PELOTON_MAX_HP
            gas_ratio = pel['gas'] / MAX_GAS
            ammo_ratio = pel['ammo'] / MAX_AMMO
            tanks_ratio = pel['num_tanks'] / TANKS_PER_PELOTON
            point_captured = [1.0 if self.captured_points[p] else 0.0 for p in ['A','B','C']]
            status = np.array([hp_ratio, gas_ratio, ammo_ratio, tanks_ratio] + point_captured, dtype=np.float32)
            
            obs_list.append({'grid': grid_window, 'status': status})
        
        return tuple(obs_list)  # Tuple para spaces.Tuple
    
    def _get_info(self):
        """
        Información adicional para depuración.
        """
        return {
            'step': self.step_count,
            'blue_alive': sum(1 for p in self.blue_pelotons if p['num_tanks'] > 0),
            'red_alive': sum(1 for p in self.red_pelotons if p['num_tanks'] > 0),
            'captured': self.captured_points.copy()
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
            print("Pygame no instalado. No se puede renderizar.")
            return
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((800, 800))
            self.clock = pygame.time.Clock()
        
        # Limpiar pantalla
        self.window.fill((0,0,0))
        cell_size = 800 // MAP_SIZE
        
        # Dibujar mapa (colores según tipo)
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
                cell = self.grid[y][x]
                if cell['type'] == 'OPEN':
                    color = (100,200,100)
                elif cell['type'] == 'BUSH':
                    color = (50,150,50)
                elif cell['type'] == 'FOREST':
                    color = (20,100,20)
                elif cell['type'] == 'RUBBLE':
                    color = (100,100,100)
                elif cell['type'] == 'WALL':
                    color = (80,80,80)
                elif cell['type'] == 'WATER':
                    color = (50,100,200)
                else:
                    color = (200,200,200)
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0,0,0), rect, 1)
        
        # Dibujar puntos A,B,C
        for name, p in self.points.items():
            if not self.captured_points[name]:
                cx = p['x']*cell_size + cell_size//2
                cy = p['y']*cell_size + cell_size//2
                pygame.draw.circle(self.window, (255,255,0), (cx,cy), cell_size//3)
        
        # Dibujar pelotones azules (cuadrados azules)
        for pel in self.blue_pelotons:
            if pel['num_tanks'] > 0:
                rect = pygame.Rect(pel['x']*cell_size, pel['y']*cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, (0,0,255), rect)
                # Mostrar número de tanques
                font = pygame.font.Font(None, 20)
                text = font.render(str(pel['num_tanks']), True, (255,255,255))
                self.window.blit(text, (pel['x']*cell_size, pel['y']*cell_size))
        
        # Dibujar rojos (cuadrados rojos)
        for pel in self.red_pelotons:
            if pel['num_tanks'] > 0:
                rect = pygame.Rect(pel['x']*cell_size, pel['y']*cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, (255,0,0), rect)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
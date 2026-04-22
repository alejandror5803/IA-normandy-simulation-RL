# implements libraries
import random
from collections import deque
import env.env_config as cfg

# Terrain types: cover value and movement penalization
TERRAIN_TYPES = {
    "OPEN":   {"cover": 0.0, "penalization": 0},
    "BUSH":   {"cover": 0.3, "penalization": 1},
    "FOREST": {"cover": 0.6, "penalization": 2},
    "RUBBLE": {"cover": 0.5, "penalization": 1},
    "WALL":   {"cover": 0.9, "penalization": 3},
    "WATER":  {"cover": 0.0, "penalization": 99},
}

# terrain types that block movement
IMPASSABLE = {"WATER", "WALL"}

HALF = cfg.MAP_SIZE // 2
# Fixed positions of the capture points (col, row)
FIXED_POINTS = {
    "A": (4,  HALF),   # West flank
    "B": (HALF, HALF),   # Center (most valuable)
    "C": ( cfg.MAP_SIZE - 4, HALF),   # East flank
}

# Maximum supplies available at each capture point
POINT_SUPPLY_LIMITS = {
    "gas":  1000,
    "ammo": 50,
}


class MapGenerator:

    # initializes the class in seed 42 , with a map size of 25
    def __init__(self, size=25, seed=42):
        self.size = size
        random.seed(seed)

    # return the terrain type
    def _cell(self, terrain):
        return {"type": terrain, **TERRAIN_TYPES[terrain]}

    # BFS expansion from a seed point — creates organic blobs of terrain
    def _spread_terrain(self, grid, start_x, start_y, terrain, max_cells, spread_prob=0.60):
        if not (0 <= start_x < self.size and 0 <= start_y < self.size):
            return

        cells_placed = 0
        queue = deque()
        visited = {(start_x, start_y)}

        grid[start_y][start_x] = self._cell(terrain)
        queue.append((start_x, start_y))
        cells_placed = 1

        while queue and cells_placed < max_cells:

            cx, cy = queue.popleft()
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            random.shuffle(dirs)

            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy

                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                visited.add((nx, ny))
                if random.random() < spread_prob:
                    grid[ny][nx] = self._cell(terrain)
                    queue.append((nx, ny))
                    cells_placed += 1
                    if cells_placed >= max_cells:
                        break

    # Place a small city-block area: WALL buildings with OPEN streets between them
    def _place_urban(self, grid, start_x, start_y, width, height):

        for row_i in range(height):
            for col_i in range(width):
                x = start_x + col_i
                y = start_y + row_i
                if not (0 <= x < self.size and 0 <= y < self.size):
                    continue
                if grid[y][x]["type"] == "WATER":
                    continue

                # streets every 3rd column and every 3rd row, rest is buildings
                if col_i % 3 == 2 or row_i % 3 == 2:
                    t = "OPEN"
                else:
                    t = "RUBBLE" if random.random() < 0.15 else "WALL"
                grid[y][x] = self._cell(t)

    def generate_map(self):
        # everything starts as open field
        grid = [[self._cell("OPEN") for _ in range(self.size)] for _ in range(self.size)]

        # lakes: 1-3 organic water bodies spread from random seed points
        num_lakes = random.randint(1, 3)
        for _ in range(num_lakes):
            sx = random.randint(2, self.size - 3)
            sy = random.randint(2, self.size - 3)
            self._spread_terrain(grid, sx, sy, "WATER", random.randint(8, 18), spread_prob=0.55)

        # forests: 2-4 forest patches
        num_forests = random.randint(2, 4)
        for _ in range(num_forests):
            sx = random.randint(1, self.size - 2)
            sy = random.randint(1, self.size - 2)
            self._spread_terrain(grid, sx, sy, "FOREST", random.randint(12, 25), spread_prob=0.60)

        # bush ring around every forest cell as a transition zone
        bush_candidates = set()
        for row_i in range(self.size):
            for col_i in range(self.size):
                if grid[row_i][col_i]["type"] == "FOREST":
                    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                        nx, ny = col_i + dx, row_i + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            if grid[ny][nx]["type"] == "OPEN":
                                bush_candidates.add((nx, ny))
        for bx, by in bush_candidates:
            grid[by][bx] = self._cell("BUSH")

        # urban zones: 1-2 small city-block areas
        num_urban = random.randint(1, 2)
        for _ in range(num_urban):
            ux = random.randint(3, self.size - 10)
            uy = random.randint(3, self.size - 10)
            self._place_urban(grid, ux, uy, random.randint(5, 8), random.randint(5, 8))

        # scatter some rubble in open areas (feels more like a battlefield)
        for row_i in range(self.size):
            for col_i in range(self.size):
                if grid[row_i][col_i]["type"] == "OPEN" and random.random() < 0.04:
                    grid[row_i][col_i] = self._cell("RUBBLE")

        # always make capture points A, B, C passable and clear their immediate neighbors
        for col, row in FIXED_POINTS.values():
            grid[row][col] = self._cell("OPEN")
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = col + dx, row + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if grid[ny][nx]["type"] in IMPASSABLE:
                        grid[ny][nx] = self._cell("OPEN")

        return grid

    # it get's fix coordinates of one spot
    @staticmethod
    def get_points():
        return FIXED_POINTS

    # verification of a specific position in passable (transitable is bool)
    @staticmethod
    def is_passable(grid, x, y):
        if not (0 <= x < len(grid[0]) and 0 <= y < len(grid)):
            return False
        return grid[y][x]["type"] not in IMPASSABLE

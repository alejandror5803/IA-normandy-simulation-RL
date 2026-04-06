# Generates the terrain and the points of interest of the Normandy map

import random

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

# accumulated probability thresholds for random terrain generation
TERRAIN_THRESHOLDS = [
    (0.10, "WATER"),
    (0.20, "WALL"),
    (0.35, "FOREST"),
    (0.50, "BUSH"),
    (0.60, "RUBBLE"),
    (1.00, "OPEN"),
]

# Fixed positions of the capture points (col, row)
FIXED_POINTS = {
    "A": (2,  12),   # West flank
    "B": (12, 12),   # Center (most valuable point)
    "C": (22, 12),   # East flank
}

# Maximum supplies available at each capture point
POINT_SUPPLY_LIMITS = {
    "gas":  1000,
    "ammo": 50,
}


class MapGenerator:

    def __init__(self, size=25, seed=42):
        self.size = size
        random.seed(seed)

    def _pick_terrain(self):
        rand = random.random()
        for threshold, terrain in TERRAIN_THRESHOLDS:
            if rand < threshold:
                return terrain
        return "OPEN"

    def generate_map(self):
        grid = []
        for _ in range(self.size):
            row = []
            for _ in range(self.size):
                t = self._pick_terrain()
                row.append({"type": t, **TERRAIN_TYPES[t]})
            grid.append(row)

        # Ensure A, B, C are never blocked by impassable terrain
        for col, row in FIXED_POINTS.values():
            grid[row][col] = {"type": "OPEN", **TERRAIN_TYPES["OPEN"]}

        return grid

    @staticmethod
    def get_points() -> dict[str, tuple[int, int]]:
        """Returns the positions of the capture points"""
        return FIXED_POINTS

    @staticmethod
    def is_passable(grid, x, y):
        if not (0 <= x < len(grid[0]) and 0 <= y < len(grid)):
            return False
        return grid[y][x]["type"] not in IMPASSABLE

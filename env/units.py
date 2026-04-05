from typing import Tuple

Position = Tuple[int, int]


'''
How much costs going throw each cell. Tanks can't go throw the water
'''

TERRAIN = {
    "OPEN":  {"cover": 0.0, "cost": 1},
    "BUSH":  {"cover": 0.3, "cost": 2},
    "FOREST":{"cover": 0.6, "cost": 3},
    "WALL":  {"cover": 0.9, "cost": 4},
    "WATER": {"cover": 0.0, "cost": 999}, 
}


'''
It defines how to measure distances. It says to the agents if they can attack
or if they can find enemies.
'''

def distance(a: Position, b: Position):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


'''
Class Platoon
Atributes:
    - team : blue or read
    - position: where it is
    - hp : total life
    - fuel: remaining fuel
    - ammo: remaining ammunition
'''

class Platoon:
    def __init__(self, team: str, position: Position, hp: int = 500,
                 fuel: int = 100, ammo: int = 50):
        self.team = team
        self.position = position
        self.hp = hp
        self.fuel = fuel
        self.ammo = ammo

    def move(self, new_pos: Position, terrain: str):
        cost = TERRAIN[terrain]["cost"]

        if cost >= 999 or self.fuel < cost:
            return False

        self.position = new_pos
        self.fuel -= cost
        return True


    def attack(self, enemy: "Platoon"):
        if self.ammo <= 0:
            return 0

        if distance(self.position, enemy.position) > 3:
            return 0

        damage = 20
        self.ammo -= 1
        enemy.hp -= damage

        if enemy.hp < 0:
            enemy.hp = 0

        return damage

    def is_alive(self):
        return self.hp > 0

    def resupply(self):
        self.fuel = min(100, self.fuel + 20)
        self.ammo = min(50, self.ammo + 5)

    def get_state(self):
        return (
            self.hp // 100,
            self.fuel // 20,
            self.ammo // 10,
        )
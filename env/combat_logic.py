# implement libraries
import env.env_config as efg
from env.units import distance
from env.map_generator import IMPASSABLE

# implementing from the enviroment constants
ATTACK_RANGE    = efg.ATTACK_RANGE
TIGER_DAMAGE    = efg.TIGER_DAMAGE   # Tiger tank — historically superior firepower
SHERMAN_DAMAGE  = efg.SHERMAN_DAMAGE   # Sherman tank — baseline
DAMAGE_PER_ATTACK = efg.DAMAGE_PER_ATTACK  # default kept for red team


def get_enemies_in_range(peloton, all_enemies, attack_range=ATTACK_RANGE):
    """ 
    Calculates the distance of each of the enemies
    all_enemies: list of all the enemies
    attack range: constant

    returns the distance of all the enemies
    ( It does not calculate the pelloton enemy when )
    """
   
    in_range = []
    for enemy in all_enemies:
        if enemy['num_tanks'] <= 0:
            continue
        if distance(peloton['pos'], enemy['pos']) <= attack_range:
            in_range.append(enemy)
    return in_range


def get_nearest_enemy(peloton, all_enemies):
    """
    Calculates the nearest enemy and their distances

    peloton: exact peloton to calculate the distance
    all_enemies: list with them
    
    returns the enemy and its distance
    """
    nearest = None
    min_dist = 99999
    for enemy in all_enemies:
        if enemy['num_tanks'] <= 0:
            continue
        d = distance(peloton['pos'], enemy['pos'])
        if d < min_dist:
            min_dist = d
            nearest = enemy
    return nearest, min_dist


def do_attack(attacker, target, target_cover=0.0, damage_per_tank=DAMAGE_PER_ATTACK):
    """
    Function that activates an attack to an target

    attacker
    target: to which enemy it shoots

    returns the damage made by the attacker
    """
    if attacker['ammo'] <= 0 or attacker['num_tanks'] <= 0:
        return 0

    base_damage = damage_per_tank * attacker['num_tanks']
    actual_damage = int(base_damage * (1.0 - target_cover))

    attacker['ammo'] -= 1
    target['hp'] -= actual_damage
    if target['hp'] < 0:
        target['hp'] = 0

    # recalculate remaining tanks from hp (each tank has 100hp)
    target['num_tanks'] = target['hp'] // 100

    return actual_damage


def get_cover_type_int(cover_value):
    
    # maps terrain cover float to the 3 levels used by defense_agent
    # 0 = no cover, 1 = light cover (bush/rubble), 2 = heavy cover (forest/wall)
    if cover_value <= 0.0:
        return 0
    elif cover_value <= 0.5:
        return 1
    else:
        return 2


def get_best_cover_cell(peloton, grid, map_size):
    # looks at the 4 adjacent cells and returns the one with the best cover
    # returns None if no adjacent cell is better than current
    px, py = peloton['pos']
    current_cover = grid[py][px]['cover']
    best_cover = current_cover
    best_pos = None

    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        nx, ny = px + dx, py + dy
        if 0 <= nx < map_size and 0 <= ny < map_size:
            cell = grid[ny][nx]
            if cell['type'] not in IMPASSABLE and cell['cover'] > best_cover:
                best_cover = cell['cover']
                best_pos = (nx, ny)

    return best_pos

# It makes a resupply of ammo or fuel to a peloton
def do_resupply(peloton, point_data):
    fuel_needed = 100 - peloton['fuel']
    ammo_needed = 100 - peloton['ammo']

    fuel_given = min(fuel_needed, point_data['gas'])
    ammo_given = min(ammo_needed, point_data['ammo'])

    peloton['fuel'] += fuel_given
    peloton['ammo'] += ammo_given
    point_data['gas']  -= fuel_given
    point_data['ammo'] -= ammo_given

    return (fuel_given + ammo_given) > 0

# it asks if all pelotons are dead
def all_dead(pelotons):
    return all(p['num_tanks'] <= 0 for p in pelotons)

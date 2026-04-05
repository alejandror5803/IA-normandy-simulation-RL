from units import distance

'''
It returns the closest enemy. If there is not enemies alive, returns None.
'''
def get_closest_enemy(platoon, enemies):
    closest_enemy = None
    min_dist = 999999

    for enemy in enemies:
        if enemy.is_alive():
            d = distance(platoon.position, enemy.position)
            if d < min_dist:
                min_dist = d
                closest_enemy = enemy

    return closest_enemy


'''
It checks if an enemy is in range.
'''
def enemy_in_range(platoon, enemy, attack_range=3):
    if enemy is None:
        return False

    return distance(platoon.position, enemy.position) <= attack_range

'''
If the closest enemy is in range, it attacks.
It returns:
    - reward
    - damage
    - target
'''
def attack_closest_enemy(platoon, enemies, attack_range=3):
    target = get_closest_enemy(platoon, enemies)

    if target is None:
        return -5, 0, None

    if not enemy_in_range(platoon, target, attack_range):
        return -2, 0, target

    old_hp = target.hp
    damage = platoon.attack(target)

    if damage == 0:
        return -1, 0, target

    reward = 10

    # Recompensa extra si destruye al enemigo
    if old_hp > 0 and not target.is_alive():
        reward += 50

    return reward, damage, target

'''
Returns how many plattons are still alive
'''
def count_alive_platoons(platoons):
    count = 0
    for platoon in platoons:
        if platoon.is_alive():
            count += 1
    return count


'''
Returns True if there are no platoons alive
'''
def team_defeated(platoons):
    return count_alive_platoons(platoons) == 0

'''
It returns a tuple with the platoon hp, the remaining fuel and ammunation,
and the distance to the closest enemy.
'''
def get_combat_state(platoon, enemies):
    
    own_state = platoon.get_state()
    closest_enemy = get_closest_enemy(platoon, enemies)

    if closest_enemy is None:
        enemy_dist = 10
    else:
        enemy_dist = min(distance(platoon.position, closest_enemy.position), 10)

    return own_state + (enemy_dist,)

'''
It decides if it is worth it to attack
'''
def should_attack(platoon, enemies, attack_range=3):
    target = get_closest_enemy(platoon, enemies)

    if target is None:
        return False

    if not platoon.is_alive():
        return False

    if platoon.ammo <= 0:
        return False

    if not enemy_in_range(platoon, target, attack_range):
        return False

    return True
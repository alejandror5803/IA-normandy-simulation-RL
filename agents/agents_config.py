

# ----------------- capture agent configuration -----------------
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
STAY = 4

# ----------------- attack agent configuration -----------------
DONT_SHOOT = 0
SHOOT = 1

# ----------------- command agent configuration -----------------
META_CAPTURE  = 0
META_ATTACK   = 1
META_DEFENSE  = 2
META_RESUPPLY = 3
ATTACK_RANGE = 3 # same as ATTACK_RANGE in combat_logic.py

# -------------- defense agent configuration -----------------
# cover types (only 3 to test)
NO_COVER = 0
BUSH = 1
WALL = 2

# actions
DONT_COVER = 0
TAKE_COVER = 1

# how much each cover type helps
COVER_BONUS = {NO_COVER: 0.0, BUSH: 0.5, WALL: 1.0}


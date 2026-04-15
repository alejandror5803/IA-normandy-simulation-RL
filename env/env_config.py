# -------- combat_logic ----------
ATTACK_RANGE    = 3
TIGER_DAMAGE    = 30   # Tiger tank — historically superior firepower
SHERMAN_DAMAGE  = 20   # Sherman tank — baseline
DAMAGE_PER_ATTACK = SHERMAN_DAMAGE  # default kept for red team

# -------- normandy_env -----------
NUM_BLUE = 4
NUM_RED  = 12
MAP_SIZE = 25
MAX_STEPS = 500 # X LO eliminamos porque lo gestiona TimeLimit

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
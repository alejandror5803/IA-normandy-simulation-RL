# -------- combat_logic ----------
ATTACK_RANGE    = 3
TIGER_DAMAGE    = 30   # Tiger tank — historically superior firepower
SHERMAN_DAMAGE  = 20   # Sherman tank — baseline
DAMAGE_PER_ATTACK = SHERMAN_DAMAGE  # default kept for red team

# -------- normandy_env -----------
NUM_BLUE = 4
NUM_RED  = 12
MAP_SIZE = 40 # 25
MAX_STEPS = 500 # X LO eliminamos porque lo gestiona TimeLimit

# commander meta-actions: the commander decides WHICH sub-agent takes action on the peloton this step
# movement direction is never chosen by the commander directly — the capture_agent handles that
META_CAPTURE  = 0   # capture_agent decides where to move
META_ATTACK   = 1   # attack_agent decides whether to shoot
META_DEFENSE  = 2   # defense_agent decides whether to seek cover
META_RESUPPLY = 3   # directly resupply at the nearest capture point

# rewards / penalties
R_CAPTURE_A_C   = 100
R_CAPTURE_B     = 300
R_DESTROY_ENEMY = 200   # higher to make killing enemies worth delegating to attack_agent
R_RESUPPLY      = 10
R_WIN           = 1000
P_LOSE          = -1000 # symmetric magnitude with R_WIN to avoid high-variance returns
P_STEP          = -1.0  # small relative to capture/win rewards so shaping signals remain visible

# observation vector size (one per blue peloton)
OBS_SIZE = 16

# seed for the map
SEED = 1 # 42

# render every N episodes default
RENDER_EVERY = 1000

# metrics / plotter
MOVING_AVG_WINDOW = 50
PLOTS_SAVE_PATH   = "results"

CAPTURE_OVERTIME = 30  # steps the enemy gets to reconquer after all 3 points are taken

PELOTON_AMMO = 25
PELOTON_FUEL = 500

TIGER_PELOTON_HP = 600
SHERMAN_PELOTON_HP = 300

# -------- combat_logic ----------
ATTACK_RANGE    = 8
TIGER_DAMAGE    = 40   # Tiger tank — historically superior firepower
SHERMAN_DAMAGE  = 20   # Sherman tank — baseline
DAMAGE_PER_ATTACK = SHERMAN_DAMAGE  # default kept for red team

# -------- normandy_env -----------
NUM_BLUE = 4
NUM_RED  = 12
MAP_SIZE = 100 # 25
MAX_STEPS = 800 # better fit for 100x100 map (movement + capture + counterplay)

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
# "enemy_nearby" is visibility-level proximity, not direct fire range
ENEMY_NEARBY_RANGE = 12

# seed for the map
SEED = 482 # 42

# render every N episodes default
RENDER_EVERY = 1000

# metrics / plotter
MOVING_AVG_WINDOW = 50
MOVING_AVG_WINDOW_LONG = 200
PLOTS_SAVE_PATH   = "results"

CAPTURE_OVERTIME = 40  # steps the enemy gets to reconquer after all 3 points are taken

PELOTON_AMMO = 25
PELOTON_FUEL = 500

TIGER_PELOTON_HP = 650
SHERMAN_PELOTON_HP = 500

# -------- Luftwaffe (German air support) ----------
LUFTWAFFE_ENABLED      = True   # toggle to disable the Luftwaffe entirely
LUFTWAFFE_BLAST_RADIUS = 5      # Manhattan-distance radius of the bomb
LUFTWAFFE_BASE_DAMAGE  = 400    # maximum damage at ground zero (before cover/falloff)
LUFTWAFFE_START_DELAY  = 50     # steps before first strike can be ordered
LUFTWAFFE_REARM_MIN_STEPS = 15  # min cooldown at base after returning
LUFTWAFFE_REARM_MAX_STEPS = 25  # max cooldown at base after returning
LUFTWAFFE_RESPAWN_STEPS = 200   # if shot down, wait this many env steps to return
LUFTWAFFE_MAX_RESPAWNS_PER_EPISODE = 1  # safety cap to avoid infinite air pressure
R_LUFTWAFFE_HIT        = 40     # reward per blue commander for each Luftwaffe hit
R_LUFTWAFFE_KILL       = 100    # extra reward when a red platoon is destroyed by air

# -------- SmolAgents Field Marshal ----------
FM_STRATEGY_INTERVAL = 1000  # episodes between LLM strategy calls
FM_WARMUP_EPISODES   = 1500  # do not call FM before this episode

# LLM provider: "hf" (HuggingFace) or "groq" (free tier, 1000 req/day, much faster)
FM_PROVIDER = "groq"
#FM_PROVIDER = "hf"

# HuggingFace provider — get token at https://huggingface.co/settings/tokens
FM_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN    = "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"

# https://console.groq.com/keys
# switch to "groq" above if HF rate-limits you
GROQ_MODEL_ID = "llama-3.3-70b-versatile"
GROQ_API_KEY  = "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"

# default reward multipliers (Option A) - LLM overrides these every N episodes
FM_DEFAULT_CAPTURE_MULT = 1.0
FM_DEFAULT_KILL_MULT    = 1.0
FM_DEFAULT_DEFEND_MULT  = 1.0

# -------- Epsilon decay tuning ----------
# Slower decay helps on this project because many agents learn in parallel
EPS_DECAY_BLUE_COMMAND  = 0.9995
EPS_DECAY_BLUE_ATTACK   = 0.9995
EPS_DECAY_BLUE_DEFENSE  = 0.9995
EPS_DECAY_BLUE_CAPTURE  = 0.9998
EPS_DECAY_RED_COMMAND   = 0.9995
EPS_DECAY_RED_ATTACK    = 0.9995
EPS_DECAY_RED_DEFENSE   = 0.9995
EPS_DECAY_RED_CAPTURE   = 0.9998

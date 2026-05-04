# Luftwaffe - German air support for the Normandy simulation
#
# Two AI techniques from the course are used here:
#
#   1. Markov Chain (Topic 5)
#      The aircraft operational cycle is modelled as a Markov Chain with 6 states:
#      AVAILABLE, INBOUND, STRIKING, RETURNING, REARMING, SHOT_DOWN
#      The transition matrix T captures the probabilities between phases.
#      There is one stochastic element: probability of being shot down by anti-aircraft
#      fire (flak) on each step while flying to the target (INBOUND state).
#
#   2. PSO - Particle Swarm Optimization (Topic 4)
#      Before each mission we use pyswarm.pso() to find the map coordinate (x, y)
#      that maximises expected damage on enemy platoons.
#      It takes into account distance to impact, terrain cover and tank count.
#      Since pso() minimises, the objective function is negated.

import io
import math
import sys
import numpy as np
import random
from pyswarm import pso


# Markov Chain state constants
AV = 0  # available at base
IB = 1  # flying toward target
ST = 2  # executing bomb run
RT = 3  # returning to base
RA = 4  # rearming
SD = 5  # shot down (absorbing state)

STATE_NAMES = {
    AV: "AVAILABLE",
    IB: "INBOUND",
    ST: "STRIKING",
    RT: "RETURNING",
    RA: "REARMING",
    SD: "SHOT_DOWN",
}


class LuftwaffeMarkovChain:
    # Models the aircraft operational lifecycle as a Markov Chain.
    # T[i][j] = P(transition to state j | currently in state i)
    #
    # Flight and rearming times are deterministic (step counters),
    # but flak interception in INBOUND is stochastic (P_FLAK per step).
    #
    # Theoretical analysis of the full cycle (no flak):
    #   cycle length = 1(AV) + FLIGHT_STEPS(IB) + 1(ST) + FLIGHT_STEPS(RT) + REARM_STEPS(RA)
    #   P(survive one mission) = (1 - P_FLAK)^FLIGHT_STEPS
    #   expected availability  ~ 1 / cycle_length

    P_FLAK       = 0.04  # prob of being shot down per step while INBOUND
    FLIGHT_STEPS = 5     # steps to reach / return from target
    REARM_STEPS  = 8     # steps to reload at base

    # Transition matrix 6x6
    # Rows / cols: AV, IB, ST, RT, RA, SD
    T = np.array([
        # AV    IB           ST          RT    RA    SD
        [0.00,  1.00,        0.00,       0.00, 0.00, 0.00],   # AV -> IB when ordered
        [0.00,  0.00,  1 - P_FLAK,       0.00, 0.00, P_FLAK], # IB -> ST or shot down
        [0.00,  0.00,        0.00,       1.00, 0.00, 0.00],   # ST -> RT always
        [0.00,  0.00,        0.00,       0.00, 1.00, 0.00],   # RT -> RA always
        [1.00,  0.00,        0.00,       0.00, 0.00, 0.00],   # RA -> AV when done
        [0.00,  0.00,        0.00,       0.00, 0.00, 1.00],   # SD absorbing
    ], dtype=np.float64)

    def __init__(self):
        self.state          = AV
        self.steps_in_state = 0
        self.target_pos     = None
        self.missions_done  = 0
        self.shot_down      = False

    def is_available(self):
        return self.state == AV and not self.shot_down

    def is_striking(self):
        return self.state == ST

    def order_strike(self, target_pos):
        # a mission can only be ordered from AVAILABLE
        if self.state != AV:
            return False
        self.target_pos     = target_pos
        self.state          = IB
        self.steps_in_state = 0
        return True

    def step(self):
        # advance one step and return (new_state, event)
        # event can be: 'STRIKING', 'SHOT_DOWN', 'AVAILABLE' or None
        if self.shot_down:
            return SD, None

        self.steps_in_state += 1
        event = None

        if self.state == IB:
            # check if anti-aircraft fire takes down the plane
            if random.random() < self.P_FLAK:
                self.state     = SD
                self.shot_down = True
                event          = "SHOT_DOWN"
            elif self.steps_in_state >= self.FLIGHT_STEPS:
                self.state          = ST
                self.steps_in_state = 0
                event               = "STRIKING"

        elif self.state == ST:
            # bomb run lasts one step, then the plane turns back
            self.state          = RT
            self.steps_in_state = 0
            self.missions_done += 1

        elif self.state == RT:
            if self.steps_in_state >= self.FLIGHT_STEPS:
                self.state          = RA
                self.steps_in_state = 0

        elif self.state == RA:
            if self.steps_in_state >= self.REARM_STEPS:
                self.state          = AV
                self.steps_in_state = 0
                event               = "AVAILABLE"

        return self.state, event

    def reset(self):
        self.state          = AV
        self.steps_in_state = 0
        self.target_pos     = None
        self.missions_done  = 0
        self.shot_down      = False

    def get_state_name(self):
        return STATE_NAMES.get(self.state, "UNKNOWN")

    @classmethod
    def get_theoretical_stats(cls):
        # computes theoretical values from the Markov Chain structure
        cycle_len         = 1 + cls.FLIGHT_STEPS + 1 + cls.FLIGHT_STEPS + cls.REARM_STEPS
        p_survive         = (1.0 - cls.P_FLAK) ** cls.FLIGHT_STEPS
        availability      = 1.0 / cycle_len
        expected_missions = 1.0 / (1.0 - p_survive) if p_survive < 1 else float("inf")
        return {
            "cycle_len":          cycle_len,
            "p_survive":          round(p_survive, 4),
            "availability":       round(availability, 4),
            "expected_missions":  round(expected_missions, 2),
        }


class LuftwaffeAgent:
    # Combines the Markov Chain (when to fly and lifecycle) with PSO (where to bomb).
    # Objective function for PSO:
    #   f(x, y) = sum of HP_enemy * (1 - cover*0.5) * distance_factor
    # Since pso() minimises, we return -f to find the maximum damage position.

    BLAST_RADIUS  = 4    # Manhattan radius of the bomb blast
    BASE_DAMAGE   = 120  # max damage at ground zero with no cover
    NUM_PARTICLES = 15   # number of PSO particles
    MAX_ITERS     = 20   # max PSO iterations
    INERTIA       = 0.5  # omega - weight of previous velocity
    C_PERSONAL    = 1.5  # phip  - pull toward personal best
    C_GLOBAL      = 1.5  # phig  - pull toward global best
    START_DELAY   = 40   # steps to wait before first strike (let spawns spread out)

    def __init__(self, map_size):
        self.map_size     = map_size
        self.mc           = LuftwaffeMarkovChain()
        self.last_target  = None
        # base is at the top-right corner (aircraft comes from Germany)
        self.base_pos     = (map_size - 1, 0)

        stats = LuftwaffeMarkovChain.get_theoretical_stats()
        print(
            f"[Luftwaffe] Markov Chain ready -- "
            f"cycle={stats['cycle_len']} steps | "
            f"P(survive mission)={stats['p_survive']:.1%} | "
            f"availability~{stats['availability']:.1%} | "
            f"expected missions~{stats['expected_missions']}"
        )

    def _objective(self, x, y, red_platoons, map_grid):
        # computes expected blast damage at (x, y), negated for PSO minimisation
        xi = int(np.clip(x, 0, self.map_size - 1))
        yi = int(np.clip(y, 0, self.map_size - 1))
        total_damage = 0.0
        for platoon in red_platoons:
            if platoon["num_tanks"] <= 0:
                continue
            dist = abs(platoon["pos"][0] - xi) + abs(platoon["pos"][1] - yi)
            if dist <= self.BLAST_RADIUS:
                cover        = map_grid[platoon["pos"][1]][platoon["pos"][0]].get("cover", 0.0)
                dist_factor  = 1.0 - dist / (self.BLAST_RADIUS + 1)
                total_damage += platoon["hp"] * (1.0 - cover * 0.5) * dist_factor * (platoon["num_tanks"] / 3.0)
        return -total_damage

    def _find_target_pso(self, red_platoons, map_grid):
        # use pyswarm to find the best strike coordinate on the map
        if not any(p["num_tanks"] > 0 for p in red_platoons):
            return None

        lb = [0.0,                 0.0                ]
        ub = [self.map_size - 1.0, self.map_size - 1.0]

        # pyswarm expects a function that takes a 1D array of parameters
        def f_obj(coords):
            return self._objective(coords[0], coords[1], red_platoons, map_grid)

        # suppress pyswarm internal prints so they don't flood the training log
        silent = io.StringIO()
        prev_stdout = sys.stdout
        sys.stdout  = silent
        try:
            xopt, fopt = pso(
                f_obj,
                lb, ub,
                swarmsize = self.NUM_PARTICLES,
                maxiter   = self.MAX_ITERS,
                omega     = self.INERTIA,
                phip      = self.C_PERSONAL,
                phig      = self.C_GLOBAL,
                debug     = False,
            )
        finally:
            sys.stdout = prev_stdout

        # if best value >= 0 there are no enemies worth striking
        if fopt >= 0:
            return None
        return (int(round(xopt[0])), int(round(xopt[1])))

    def try_order_strike(self, red_platoons, map_grid, current_step=0):
        # order a mission if the plane is available and the initial delay has passed
        if current_step < self.START_DELAY:
            return False
        if not self.mc.is_available():
            return False
        target = self._find_target_pso(red_platoons, map_grid)
        if target is None:
            return False
        self.last_target = target
        return self.mc.order_strike(target)

    def step(self, red_platoons, map_grid):
        # advance the lifecycle by one step and execute the strike if it fires
        # returns a list of hit records for that step
        _, event = self.mc.step()

        hits = []
        if event == "STRIKING" and self.last_target is not None:
            tx, ty = self.last_target
            for platoon in red_platoons:
                if platoon["num_tanks"] <= 0:
                    continue
                dist = abs(platoon["pos"][0] - tx) + abs(platoon["pos"][1] - ty)
                if dist <= self.BLAST_RADIUS:
                    cover       = map_grid[platoon["pos"][1]][platoon["pos"][0]].get("cover", 0.0)
                    dist_factor = 1.0 - dist / (self.BLAST_RADIUS + 1)
                    dmg         = int(self.BASE_DAMAGE * dist_factor * (1.0 - cover * 0.5))
                    if dmg > 0:
                        prev_tanks           = platoon["num_tanks"]
                        platoon["hp"]       -= dmg
                        platoon["hp"]        = max(0, platoon["hp"])
                        platoon["num_tanks"] = platoon["hp"] // 100
                        hits.append({"platoon": platoon, "damage": dmg, "old_tanks": prev_tanks})
        return hits

    def reset(self):
        self.mc.reset()
        self.last_target = None

    def get_visual_pos(self):
        # returns the float (x, y) grid position of the plane for rendering
        # linearly interpolates between base and target based on current state
        mc = self.mc
        if self.last_target is None:
            return None

        bx, by = float(self.base_pos[0]), float(self.base_pos[1])
        tx, ty = float(self.last_target[0]), float(self.last_target[1])

        if mc.state == IB:
            t = min(1.0, mc.steps_in_state / max(1, mc.FLIGHT_STEPS))
            return (bx + (tx - bx) * t, by + (ty - by) * t)
        if mc.state == ST:
            return (tx, ty)
        if mc.state == RT:
            t = min(1.0, mc.steps_in_state / max(1, mc.FLIGHT_STEPS))
            return (tx + (bx - tx) * t, ty + (by - ty) * t)
        return None

    def get_flight_direction(self):
        # returns the normalised (dx, dy) direction vector to rotate the sprite
        mc = self.mc
        if self.last_target is None:
            return (0.0, 0.0)

        bx, by = float(self.base_pos[0]), float(self.base_pos[1])
        tx, ty = float(self.last_target[0]), float(self.last_target[1])

        if mc.state == IB:
            dx, dy = tx - bx, ty - by
        elif mc.state == RT:
            dx, dy = bx - tx, by - ty
        else:
            return (0.0, -1.0)

        length = math.hypot(dx, dy)
        if length < 1e-6:
            return (0.0, -1.0)
        return (dx / length, dy / length)

    @property
    def state_name(self):
        return self.mc.get_state_name()

    @property
    def is_active(self):
        return not self.mc.shot_down

    @property
    def current_target(self):
        return self.mc.target_pos if self.mc.state == IB else None

    @property
    def missions_done(self):
        return self.mc.missions_done

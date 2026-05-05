from env.normandy_env import make_env
from agents.command_agent import command_agent
from agents.field_marshal import SmolAgentsFieldMarshal, AlliedFieldMarshal
from utils.metrics_and_plotter import EpisodeTracker, plot_all
import env.env_config as cfg

# meta-action index to FM multiplier key (blue only)
_MULT_KEY = {
    cfg.META_CAPTURE:  "capture_mult",
    cfg.META_ATTACK:   "kill_mult",
    cfg.META_DEFENSE:  "defend_mult",
    cfg.META_RESUPPLY: None,
}


def _build_blue_fm_stats(tracker, base_env, commanders) -> dict:
    # compile blue team stats for the German Field Marshal
    window = min(cfg.MOVING_AVG_WINDOW, len(tracker.total_rewards))
    if window == 0:
        return {}

    recent_rewards  = tracker.total_rewards[-window:]
    recent_blue     = tracker.blue_alive_end[-window:]
    recent_caps     = tracker.blue_captures_end[-window:]
    # use winner field which covers all 3 win conditions (kill, capture overtime, time limit)
    recent_winners  = tracker.winner_end[-window:]
    wins            = sum(1 for w in recent_winners if w == 'blue')

    return {
        "avg_reward":     sum(recent_rewards) / window,
        "win_rate":       wins / window,
        "avg_captures":   sum(recent_caps) / window,
        "blue_alive_avg": sum(recent_blue) / window,
        "blue_epsilons":  [round(c.epsilon, 3) for c in commanders],
        "blue_q_means":   [round(float(c.q_table.mean()), 4) for c in commanders],
        "lw_missions":    base_env.luftwaffe_agent.missions_done if base_env.luftwaffe_agent else 0,
        "lw_active":      base_env.luftwaffe_agent.is_active     if base_env.luftwaffe_agent else False,
    }


def _build_red_fm_stats(tracker, base_env) -> dict:
    # compile red team stats for the Allied Field Marshal
    window = min(cfg.MOVING_AVG_WINDOW, len(tracker.total_rewards))
    if window == 0:
        return {}

    recent_blue    = tracker.blue_alive_end[-window:]
    recent_red     = tracker.red_alive_end[-window:]
    recent_winners = tracker.winner_end[-window:]
    red_wins       = sum(1 for w in recent_winners if w == 'red')

    squad_stats = base_env.get_red_squad_stats()

    return {
        "red_win_rate":      red_wins / window,
        "avg_red_alive":     sum(recent_red) / window,
        "avg_blue_alive":    sum(recent_blue) / window,
        "red_group_eps":     [s["eps"]    for s in squad_stats],
        "red_group_qmeans":  [s["q_mean"] for s in squad_stats],
    }


def train(episodes=5000, render_every=1000):
    env = make_env(render_mode="human", render_every=render_every)

    tracker = EpisodeTracker()

    # one independent commander per blue platoon
    commanders = [command_agent() for _ in range(4)]

    # --- German Field Marshal (blue side) ---
    # Option A: reward multipliers for capture / kill / defend
    # Option B: individual epsilon for each of the 4 blue commanders
    blue_fm        = SmolAgentsFieldMarshal()
    blue_strategy  = blue_fm.current_strategy

    # --- Allied Field Marshal (red side) ---
    # Option B only: individual epsilon for each of the 4 red squads (3 commanders each)
    red_fm         = AlliedFieldMarshal()
    red_strategy   = red_fm.current_strategy

    fm_next_call   = cfg.FM_STRATEGY_INTERVAL

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done         = False
        ep_td_errors = []
        env.unwrapped.increase_episode()

        while not done:
            actions = [commanders[i].choose_action(obs[i]) for i in range(4)]

            obs_new, rewards, terminated, truncated, info = env.step(actions)

            for i in range(4):
                # Option A (blue): pick the FM reward multiplier for the action taken
                mult_key = _MULT_KEY.get(actions[i])
                mult     = blue_strategy.get(mult_key, 1.0) if mult_key else 1.0

                # strip P_STEP before scaling - we don't want the step penalty amplified
                base_r = rewards[i] - cfg.P_STEP
                r = commanders[i].compute_reward(
                    obs[i], obs_new[i], actions[i], base_r,
                    reward_mult=mult,
                )
                r += cfg.P_STEP

                td_err = commanders[i].update(obs[i], actions[i], r, obs_new[i])
                ep_td_errors.append(td_err)

            obs           = obs_new
            total_reward += sum(rewards)
            done          = terminated or truncated

        for commander in commanders:
            commander.decay_epsilon(decay_rate=0.999, min_epsilon=0.05)

        cmd_td_mean = sum(ep_td_errors) / len(ep_td_errors) if ep_td_errors else 0.0
        tracker.record(total_reward, info['step'], cmd_td_mean, info, commanders, env.unwrapped)

        # both Field Marshals are called at the same interval
        if (ep + 1) >= fm_next_call:
            base_env = env.unwrapped

            # --- German FM update ---
            blue_stats   = _build_blue_fm_stats(tracker, base_env, commanders)
            blue_strategy = blue_fm.update_strategy(blue_stats)

            # Option A: reward multipliers for blue (red stays neutral)
            base_env.set_fm_reward_mults(
                capture=blue_strategy.get("capture_mult", 1.0),
                kill=blue_strategy.get("kill_mult",       1.0),
                defend=blue_strategy.get("defend_mult",   1.0),
            )
            # Option B: individual epsilon per blue commander
            new_blue_eps = blue_strategy.get("blue_eps", [])
            if len(new_blue_eps) == len(commanders):
                for i, eps in enumerate(new_blue_eps):
                    commanders[i].epsilon = eps

            # --- Allied FM update ---
            red_stats   = _build_red_fm_stats(tracker, base_env)
            red_strategy = red_fm.update_strategy(red_stats)

            # Option B: individual epsilon per red squad (4 squads x 3 commanders)
            new_red_eps = red_strategy.get("red_group_eps", [])
            if len(new_red_eps) == 4:
                base_env.set_red_epsilons(new_red_eps)

            fm_next_call += cfg.FM_STRATEGY_INTERVAL

        if (ep + 1) % 50 == 0:
            avg_window = sum(tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:]) / len(
                tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:]
            )
            blue_caps = sum(1 for v in info['captured'].values() if v)
            red_caps  = sum(1 for v in info['red_captured'].values() if v)
            lw_state  = info.get('lw_state', 'N/A')
            lw_miss   = info.get('lw_missions', 0)
            cm = blue_strategy.get('capture_mult', 1.0)
            km = blue_strategy.get('kill_mult',    1.0)
            dm = blue_strategy.get('defend_mult',  1.0)
            print(
                f"ep {ep + 1:4d}  "
                f"reward={total_reward:8.1f}  "
                f"avg{cfg.MOVING_AVG_WINDOW}={avg_window:8.1f}  "
                f"blue={info['blue_alive']}  "
                f"red={info['red_alive']}  "
                f"blue_caps={blue_caps}/3  "
                f"red_caps={red_caps}/3  "
                f"blue_eps={commanders[0].epsilon:.3f}  "
                f"red_eps={info['red_eps']:.3f}  "
                f"lw={lw_state}({lw_miss})  "
                f"fm_mults=c{cm:.1f}/k{km:.1f}/d{dm:.1f}  "
                f"src_b={blue_strategy.get('source','?')}  "
                f"src_r={red_strategy.get('source','?')}"
            )

    base_env = env.unwrapped
    env.close()
    plot_all(tracker, commanders, base_env)


if __name__ == "__main__":
    train(episodes=10000, render_every=500)

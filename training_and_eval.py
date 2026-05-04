from env.normandy_env import make_env
from agents.command_agent import command_agent
from agents.field_marshal import SmolAgentsFieldMarshal
from utils.metrics_and_plotter import EpisodeTracker, plot_all
import env.env_config as cfg


def _build_fm_stats(tracker, ep, base_env) -> dict:
    """Compile recent battle statistics for the Field Marshal LLM call."""
    window = min(cfg.MOVING_AVG_WINDOW, len(tracker.total_rewards))
    if window == 0:
        return {}

    recent_rewards   = tracker.total_rewards[-window:]
    recent_blue      = tracker.blue_alive_end[-window:]
    recent_caps      = tracker.blue_captures_end[-window:]
    recent_red       = tracker.red_alive_end[-window:]

    # Wins = episodes where more red died than blue (simple heuristic)
    wins = sum(
        1 for b, r in zip(recent_blue, recent_red) if b > 0 and r == 0
    )

    return {
        "avg_reward":    sum(recent_rewards) / window,
        "win_rate":      wins / window,
        "avg_captures":  sum(recent_caps) / window,
        "blue_alive_avg": sum(recent_blue) / window,
        "lw_missions":   base_env.luftwaffe_agent.missions_done
                         if base_env.luftwaffe_agent else 0,
        "lw_active":     base_env.luftwaffe_agent.is_active
                         if base_env.luftwaffe_agent else False,
    }


def train(episodes=5000, render_every=1000):
    env = make_env(render_mode="human", render_every=render_every)

    tracker = EpisodeTracker()

    # One independent commander per blue platoon
    commanders = [command_agent() for _ in range(4)]

    # Field Marshal — LLM strategic advisor (smolagents, Tema 3).
    # Called every FM_STRATEGY_INTERVAL episodes; falls back to rule-based
    # automatically if smolagents is not installed or the API is unavailable.
    field_marshal  = SmolAgentsFieldMarshal()
    fm_strategy    = field_marshal.current_strategy   # initial default directive
    fm_next_call   = cfg.FM_STRATEGY_INTERVAL         # episode counter for next call

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
                r = commanders[i].compute_reward(
                    obs[i], obs_new[i], actions[i], rewards[i]
                )

                # ── Field Marshal strategic bonus ─────────────────────────────
                # Add a small shaped reward when the commander follows the FM
                # priority action.  Keeps the LLM in the learning signal loop
                # without taking over from Q-Learning.
                if actions[i] == fm_strategy.get("priority_action", -1):
                    r += cfg.FM_STRATEGY_BONUS

                td_err = commanders[i].update(obs[i], actions[i], r, obs_new[i])
                ep_td_errors.append(td_err)

            obs           = obs_new
            total_reward += sum(rewards)
            done          = terminated or truncated

        for commander in commanders:
            commander.decay_epsilon(decay_rate=0.999, min_epsilon=0.05)

        cmd_td_mean = sum(ep_td_errors) / len(ep_td_errors) if ep_td_errors else 0.0
        tracker.record(total_reward, info['step'], cmd_td_mean, info, commanders, env.unwrapped)

        # ── Field Marshal LLM call every N episodes ───────────────────────────
        if (ep + 1) >= fm_next_call:
            base_env   = env.unwrapped
            fm_stats   = _build_fm_stats(tracker, ep, base_env)
            fm_strategy = field_marshal.update_strategy(fm_stats)
            fm_next_call += cfg.FM_STRATEGY_INTERVAL

        if (ep + 1) % 50 == 0:
            avg_window = sum(tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:]) / len(
                tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:]
            )
            blue_caps = sum(1 for v in info['captured'].values() if v)
            red_caps  = sum(1 for v in info['red_captured'].values() if v)
            lw_state  = info.get('lw_state', 'N/A')
            lw_miss   = info.get('lw_missions', 0)
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
                f"fm={fm_strategy['priority_label']}"
            )

    base_env = env.unwrapped
    env.close()
    plot_all(tracker, commanders, base_env)


if __name__ == "__main__":
    train(episodes=10000, render_every=50)

from env.normandy_env import make_env
from agents.command_agent import command_agent
from utils.metrics_and_plotter import EpisodeTracker, plot_all
import env.env_config as cfg


def train(episodes=5000, render_every=1000):
    env = make_env(render_mode="human", render_every=render_every)

    tracker = EpisodeTracker()

    # one independent commander per blue peloton
    commanders = [command_agent() for _ in range(4)]

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        ep_td_errors = []
        env.unwrapped.increase_episode()

        while not done:
            # each commander[i] picks ONE action for its own peloton i
            actions = [commanders[i].choose_action(obs[i]) for i in range(4)]

            obs_new, rewards, terminated, truncated, info = env.step(actions)

            # each commander learns only from its own peloton experience
            for i in range(4):
                r      = commanders[i].compute_reward(obs[i], obs_new[i], actions[i], rewards[i])
                td_err = commanders[i].update(obs[i], actions[i], r, obs_new[i])
                ep_td_errors.append(td_err)

            obs = obs_new
            total_reward += sum(rewards)
            done = terminated or truncated

        for commander in commanders:
            commander.decay_epsilon(decay_rate=0.999, min_epsilon=0.05)

        cmd_td_mean = sum(ep_td_errors) / len(ep_td_errors) if ep_td_errors else 0.0
        tracker.record(total_reward, info['step'], cmd_td_mean, info, commanders, env.unwrapped)

        if (ep + 1) % 50 == 0:
            avg_window = sum(tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:]) / len(tracker.total_rewards[-cfg.MOVING_AVG_WINDOW:])
            blue_caps  = sum(1 for v in info['captured'].values() if v)
            red_caps   = sum(1 for v in info['red_captured'].values() if v)
            print(
                f"ep {ep + 1:4d}  "
                f"reward={total_reward:8.1f}  "
                f"avg{cfg.MOVING_AVG_WINDOW}={avg_window:8.1f}  "
                f"blue={info['blue_alive']}  "
                f"red={info['red_alive']}  "
                f"blue_caps={blue_caps}/3  "
                f"red_caps={red_caps}/3  "
                f"blue_eps={commanders[0].epsilon:.3f}  "
                f"red_eps={info['red_eps']:.3f}"
            )

    base_env = env.unwrapped
    env.close()
    plot_all(tracker, commanders, base_env)


if __name__ == "__main__":
    train(episodes=10000, render_every=1000)

from env.normandy_env import NormandyEnv
from agents.command_agent import command_agent


def train(episodes=5000, render_every=500):
    # env = NormandyEnv(render_mode="human")
    env = NormandyEnv(render_mode="human")
    #commander = command_agent()

    rewards_history = []   # to compute moving average

    # one independent commander per blue peloton
    commanders = [command_agent() for _ in range(4)]

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        env.increase_episode()
        

        while not done:
            # each commander[i] picks ONE action for its own peloton i
            actions = [commanders[i].choose_action(obs[i]) for i in range(4)]

            obs_new, rewards, terminated, truncated, info = env.step(actions)

            # each commander learns only from its own peloton experience
            for i in range(4):
                r = commanders[i].compute_reward(obs[i], obs_new[i], actions[i], rewards[i])
                commanders[i].update(obs[i], actions[i], r, obs_new[i])

            obs = obs_new
            total_reward += sum(rewards)
            done = terminated or truncated

        for commander in commanders:
            commander.decay_epsilon(decay_rate=0.999, min_epsilon=0.05)

        rewards_history.append(total_reward)

        if (ep + 1) % 50 == 0:
            avg50 = sum(rewards_history[-50:]) / len(rewards_history[-50:])
            caps  = sum(1 for v in info['captured'].values() if v)
            print(
                f"ep {ep + 1:4d}  "
                f"reward={total_reward:8.1f}  "
                f"avg50={avg50:8.1f}  "
                f"blue={info['blue_alive']}  "
                f"red={info['red_alive']}  "
                f"captured={caps}/3  "
                f"eps={commanders[0].epsilon:.3f}"
            )

    env.close()


if __name__ == "__main__":
    train(episodes=10000, render_every=1000)

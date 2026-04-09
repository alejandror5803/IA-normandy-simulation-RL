from env.normandy_env import NormandyEnv
from agents.command_agent import command_agent


def train(episodes=5000):
    # env = NormandyEnv(render_mode="human")
    env = NormandyEnv()
    commander = command_agent()

    rewards_history = []   # to compute moving average

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # commander picks meta-action per peloton: 0=capture, 1=attack, 2=defense, 3=resupply
            actions = commander.choose_actions_for_team(obs)
            obs_new, rewards, terminated, truncated, info = env.step(actions)

            # commander learns from every step for each peloton
            # sub-agents (attack, defense, capture) already learn inside env.step
            for i in range(4):
                shaped_reward = commander.compute_reward(obs[i], obs_new[i], actions[i], rewards[i])
                commander.update(obs[i], actions[i], shaped_reward, obs_new[i])

            obs = obs_new
            total_reward += sum(rewards)
            done = terminated or truncated

        # slower decay for the commander: ~2230 episodes to hit min
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
                f"eps={commander.epsilon:.3f}"
            )

    env.close()


if __name__ == "__main__":
    train()

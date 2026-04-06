from env.normandy_env import NormandyEnv


def train(episodes=500):
    env = NormandyEnv()

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # commander agent not implemented yet, using random actions as placeholder
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += sum(rewards)
            done = terminated or truncated

        if (ep + 1) % 50 == 0:
            caps = sum(1 for v in info['captured'].values() if v)
            print(
                f"ep {ep + 1:4d}  "
                f"reward={total_reward:8.1f}  "
                f"blue={info['blue_alive']}  "
                f"red={info['red_alive']}  "
                f"captured={caps}/3"
            )

    env.close()


if __name__ == "__main__":
    train()

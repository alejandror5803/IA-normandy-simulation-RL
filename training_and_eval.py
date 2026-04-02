import gymnasium as g
import numpy as np
from agents import command_agent, red_agent

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.observe((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
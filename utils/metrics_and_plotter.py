import os
import numpy as np
import matplotlib.pyplot as plt
import env.env_config as cfg

MOVING_AVG_WINDOW = cfg.MOVING_AVG_WINDOW
PLOTS_SAVE_PATH   = cfg.PLOTS_SAVE_PATH


class EpisodeTracker:

    def __init__(self):
        self.total_rewards   = []
        self.episode_steps   = []
        self.cmd_td_errors   = []   # mean abs TD error of blue command agents per episode

        self.blue_alive_end    = []
        self.red_alive_end     = []
        self.blue_captures_end = []
        self.red_captures_end  = []

        # epsilon at end of episode, peloton 0 as representative for each agent type
        self.blue_cmd_epsilon = []
        self.blue_atk_epsilon = []
        self.blue_def_epsilon = []
        self.blue_cap_epsilon = []
        self.red_cmd_epsilon  = []

    def record(self, total_reward, episode_steps, cmd_td_error_mean, info, commanders, env_raw):
        self.total_rewards.append(total_reward)
        self.episode_steps.append(episode_steps)
        self.cmd_td_errors.append(cmd_td_error_mean)

        self.blue_alive_end.append(info['blue_alive'])
        self.red_alive_end.append(info['red_alive'])
        self.blue_captures_end.append(sum(1 for v in info['captured'].values() if v))
        self.red_captures_end.append(sum(1 for v in info['red_captured'].values() if v))

        self.blue_cmd_epsilon.append(commanders[0].epsilon)
        self.blue_atk_epsilon.append(env_raw.attack_agents[0].epsilon)
        self.blue_def_epsilon.append(env_raw.defense_agents[0].epsilon)
        self.blue_cap_epsilon.append(env_raw.capture_agents[0].epsilon)
        self.red_cmd_epsilon.append(info['red_eps'])


def _moving_avg(values, window):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


# ============================================================
# 1.  TRAINING CURVES  (reward, duration, TD error)
# ============================================================

def plot_training_curves(tracker, save_path=PLOTS_SAVE_PATH, window=MOVING_AVG_WINDOW):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    episodes = list(range(1, len(tracker.total_rewards) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training performance', fontsize=13)

    # --- 1. reward per episode ---
    avg_rewards = _moving_avg(tracker.total_rewards, window)
    axes[0].plot(episodes, tracker.total_rewards, alpha=0.2, color='steelblue')
    axes[0].plot(episodes, avg_rewards, color='steelblue', linewidth=2,
                 label=f'moving avg ({window} ep)')
    axes[0].set_title('Reward per episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- 2. episode duration (steps) ---
    avg_steps = _moving_avg(tracker.episode_steps, window)
    axes[1].plot(episodes, tracker.episode_steps, alpha=0.2, color='darkorange')
    axes[1].plot(episodes, avg_steps, color='darkorange', linewidth=2,
                 label=f'moving avg ({window} ep)')
    axes[1].set_title('Episode duration (steps)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- 3. TD error (training error) ---
    avg_td = _moving_avg(tracker.cmd_td_errors, window)
    axes[2].plot(episodes, tracker.cmd_td_errors, alpha=0.2, color='crimson')
    axes[2].plot(episodes, avg_td, color='crimson', linewidth=2,
                 label=f'moving avg ({window} ep)')
    axes[2].set_title('Command agent — TD error')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Mean |TD error|')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'training_curves.png'), dpi=120)
    plt.show()
    plt.close(fig)

    # --- extra: epsilon decay for all agent types ---
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, tracker.blue_cmd_epsilon, label='blue command', color='navy')
    ax.plot(episodes, tracker.blue_atk_epsilon, label='blue attack',  color='crimson')
    ax.plot(episodes, tracker.blue_def_epsilon, label='blue defense', color='forestgreen')
    ax.plot(episodes, tracker.blue_cap_epsilon, label='blue capture', color='darkorange')
    ax.plot(episodes, tracker.red_cmd_epsilon,  label='red command',  color='darkred',
            linestyle='--')
    ax.set_title('Epsilon decay — all agent types (peloton 0)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig2.savefig(os.path.join(save_path, 'epsilon_decay.png'), dpi=120)
    plt.show()
    plt.close(fig2)


# ============================================================
# 2.  POLICY VISUALIZATION  (per agent type)
# ============================================================


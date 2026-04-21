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

def _command_agent_heatmaps(commander, axes_policy, axes_values):
    # state encoding: hp_level(3)*54 + threat_level(3)*18 + dx(3)*6 + dy(3)*2 + ammo(2)
    # fix dx=1 (obj to the right), dy=1 (obj below)  — a typical navigation moment
    # vary hp_level (x-axis) and threat_level (y-axis)
    # two subplots side by side: normal ammo (left) and low ammo (right)

    action_names  = ['Capture', 'Attack', 'Defense', 'Resupply']
    hp_labels     = ['Low\n(0-1)', 'Mid\n(2-3)', 'High\n(4-5)']
    threat_labels = ['No enemy', 'Enemy\nout of range', 'Enemy\nin range']

    for ammo_idx, ammo_val in enumerate([0, 1]):
        policy_grid = np.zeros((3, 3), dtype=int)
        value_grid  = np.zeros((3, 3))

        for hp in range(3):
            for threat in range(3):
                state = hp * 54 + threat * 18 + 1 * 6 + 1 * 2 + ammo_val
                q_vals = commander.q_table[state]
                policy_grid[threat][hp] = int(np.argmax(q_vals))
                value_grid[threat][hp]  = float(np.max(q_vals))

        ax_p = axes_policy[ammo_idx]
        ax_v = axes_values[ammo_idx]

        cmap_p = plt.cm.get_cmap('tab10', 4)
        ax_p.imshow(policy_grid, cmap=cmap_p, vmin=0, vmax=3, aspect='auto')
        ax_p.set_xticks(range(3))
        ax_p.set_xticklabels(hp_labels, fontsize=8)
        ax_p.set_xlabel('HP level', fontsize=8)
        ax_p.set_yticks(range(3))
        ax_p.set_yticklabels(threat_labels, fontsize=8)
        ax_p.set_ylabel('Threat level', fontsize=8)
        ax_p.set_title(f'Policy — {"low ammo" if ammo_val else "normal ammo"}', fontsize=9)
        for r in range(3):
            for c in range(3):
                ax_p.text(c, r, action_names[policy_grid[r][c]],
                          ha='center', va='center', fontsize=8, color='white',
                          bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

        im_v = ax_v.imshow(value_grid, cmap='RdYlGn', aspect='auto')
        ax_v.set_xticks(range(3))
        ax_v.set_xticklabels(hp_labels, fontsize=8)
        ax_v.set_xlabel('HP level', fontsize=8)
        ax_v.set_yticks(range(3))
        ax_v.set_yticklabels(threat_labels, fontsize=8)
        ax_v.set_ylabel('Threat level', fontsize=8)
        ax_v.set_title(f'State values — {"low ammo" if ammo_val else "normal ammo"}', fontsize=9)
        plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
        for r in range(3):
            for c in range(3):
                ax_v.text(c, r, f'{value_grid[r][c]:.1f}',
                          ha='center', va='center', fontsize=8, color='black')


def _capture_agent_heatmaps(cap_agent, axes_policy, axes_values):
    # state: dx_sign*9 + dy_sign*3 + dist_bucket
    # 3 subplots (one per dist_bucket), axes: dy_sign (y) x dx_sign (x)
    # shows: given where the objective is relative to me, which direction do I move?

    action_names = ['Up', 'Down', 'Left', 'Right', 'Stay']
    dx_labels    = ['Same col', 'Obj right', 'Obj left']
    dy_labels    = ['Same row', 'Obj below', 'Obj above']
    dist_labels  = ['Close\n(<3 tiles)', 'Medium\n(3-7 tiles)', 'Far\n(>7 tiles)']

    cmap_p = plt.cm.get_cmap('tab10', 5)

    for dist_idx in range(3):
        policy_grid = np.zeros((3, 3), dtype=int)
        value_grid  = np.zeros((3, 3))

        for dx in range(3):
            for dy in range(3):
                state = dx * 9 + dy * 3 + dist_idx
                q_vals = cap_agent.q_table[state]
                policy_grid[dy][dx] = int(np.argmax(q_vals))
                value_grid[dy][dx]  = float(np.max(q_vals))

        ax_p = axes_policy[dist_idx]
        ax_v = axes_values[dist_idx]

        ax_p.imshow(policy_grid, cmap=cmap_p, vmin=0, vmax=4, aspect='auto')
        ax_p.set_xticks(range(3))
        ax_p.set_xticklabels(dx_labels, fontsize=7, rotation=10)
        ax_p.set_xlabel('X direction to obj', fontsize=7)
        ax_p.set_yticks(range(3))
        ax_p.set_yticklabels(dy_labels, fontsize=7)
        ax_p.set_ylabel('Y direction to obj', fontsize=7)
        ax_p.set_title(f'Policy — {dist_labels[dist_idx]}', fontsize=9)
        for r in range(3):
            for c in range(3):
                ax_p.text(c, r, action_names[policy_grid[r][c]],
                          ha='center', va='center', fontsize=9, color='white',
                          bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

        im_v = ax_v.imshow(value_grid, cmap='RdYlGn', aspect='auto')
        ax_v.set_xticks(range(3))
        ax_v.set_xticklabels(dx_labels, fontsize=7, rotation=10)
        ax_v.set_xlabel('X direction to obj', fontsize=7)
        ax_v.set_yticks(range(3))
        ax_v.set_yticklabels(dy_labels, fontsize=7)
        ax_v.set_ylabel('Y direction to obj', fontsize=7)
        ax_v.set_title(f'Values — {dist_labels[dist_idx]}', fontsize=9)
        plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
        for r in range(3):
            for c in range(3):
                ax_v.text(c, r, f'{value_grid[r][c]:.1f}',
                          ha='center', va='center', fontsize=8, color='black')


def _attack_agent_bars(atk_agent, ax):
    # 2 states x 2 actions — shows if agent learned to shoot when enemy is present
    state_labels  = ['No enemy\nin range', 'Enemy\nin range']
    action_labels = ["Don't shoot", 'Shoot']
    colors        = ['#4472C4', '#ED7D31']

    x     = np.arange(2)
    width = 0.35

    for a_idx in range(2):
        q_vals = [atk_agent.q_table[s][a_idx] for s in range(2)]
        ax.bar(x + a_idx * width, q_vals, width, label=action_labels[a_idx], color=colors[a_idx])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(state_labels, fontsize=9)
    ax.set_ylabel('Q-value')
    ax.set_title('Attack agent — Q-values by state')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def _defense_agent_heatmaps(def_agent, ax_policy, ax_values):
    # state: enemy_nearby(0-1)*3 + cover_type(0-2) = 6 states
    # shows: given threat level and current cover, does the agent decide to take cover?

    cover_labels = ['No cover', 'Bush', 'Wall']
    enemy_labels = ['No enemy', 'Enemy nearby']
    action_names = ['Stay', 'Take cover']

    policy_grid = np.zeros((2, 3), dtype=int)
    value_grid  = np.zeros((2, 3))

    for enemy in range(2):
        for cover in range(3):
            state = enemy * 3 + cover
            q_vals = def_agent.q_table[state]
            policy_grid[enemy][cover] = int(np.argmax(q_vals))
            value_grid[enemy][cover]  = float(np.max(q_vals))

    cmap_p = plt.cm.get_cmap('RdYlGn', 2)
    ax_policy.imshow(policy_grid, cmap=cmap_p, vmin=0, vmax=1, aspect='auto')
    ax_policy.set_xticks(range(3))
    ax_policy.set_xticklabels(cover_labels)
    ax_policy.set_xlabel('Cover type')
    ax_policy.set_yticks(range(2))
    ax_policy.set_yticklabels(enemy_labels)
    ax_policy.set_title('Defense agent — policy')
    for r in range(2):
        for c in range(3):
            ax_policy.text(c, r, action_names[policy_grid[r][c]],
                           ha='center', va='center', fontsize=9, color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    im_v = ax_values.imshow(value_grid, cmap='RdYlGn', aspect='auto')
    ax_values.set_xticks(range(3))
    ax_values.set_xticklabels(cover_labels)
    ax_values.set_xlabel('Cover type')
    ax_values.set_yticks(range(2))
    ax_values.set_yticklabels(enemy_labels)
    ax_values.set_title('Defense agent — state values')
    plt.colorbar(im_v, ax=ax_values, fraction=0.046, pad=0.04)
    for r in range(2):
        for c in range(3):
            ax_values.text(c, r, f'{value_grid[r][c]:.1f}',
                           ha='center', va='center', fontsize=9, color='black')


def plot_agent_policies(commanders, env_raw, save_path=PLOTS_SAVE_PATH):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    commander = commanders[0]
    atk_agent = env_raw.attack_agents[0]
    def_agent = env_raw.defense_agents[0]
    cap_agent = env_raw.capture_agents[0]

    # --- command agent ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        'Command agent — policy and state values\n'
        '(axes: HP level vs Threat level  |  fixed: objective is to the right and below)',
        fontsize=11
    )
    _command_agent_heatmaps(commander, axes[0], axes[1])
    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'command_agent_policy.png'), dpi=120)
    plt.show()
    plt.close(fig)

    # --- capture agent ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        'Capture agent — policy and state values\n'
        '(axes: X and Y direction to objective  |  one column per distance range)',
        fontsize=11
    )
    _capture_agent_heatmaps(cap_agent, axes[0], axes[1])
    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'capture_agent_policy.png'), dpi=120)
    plt.show()
    plt.close(fig)

    # --- attack + defense combined ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Attack and defense agents — Q-values and policy', fontsize=11)
    _attack_agent_bars(atk_agent, axes[0])
    _defense_agent_heatmaps(def_agent, axes[1], axes[2])
    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'attack_defense_policy.png'), dpi=120)
    plt.show()
    plt.close(fig)


def plot_all(tracker, commanders, env_raw, save_path=PLOTS_SAVE_PATH, window=MOVING_AVG_WINDOW):
    plot_training_curves(tracker, save_path=save_path, window=window)
    plot_agent_policies(commanders, env_raw, save_path=save_path)

"""
Generate simulation visualization figures:
  Figure 1: Zone occupation heatmap + dispatcher action overlay (k=1, k=10, k=35)
  Figure 3: Mode-tracking accuracy curve as a function of k
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))
from alternating_marl import AlternatingMARL
from marl_example import (
    N_SG, N_SL, N_GA, N_AL, N_AGENTS, GAMMA,
    P_g, P_l, r_g, r_l,
)

_hp_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')
with open(_hp_path, 'r') as f:
    HP = json.load(f)

SAVE_DIR = os.path.dirname(os.path.dirname(__file__))


# ======================================================================
# Simulation with per-step recording
# ======================================================================

def simulate_episode(alg, horizon=50, seed=42):
    """
    Run one episode and record per-step data.

    Returns dict with arrays of shape (horizon,):
      - zone_counts: (horizon, n_sl) — agent counts per zone each step
      - dispatcher_zone: (horizon,) — zone the dispatcher steers to (= a_g)
      - true_mode: (horizon,) — zone with most agents
      - reward: (horizon,) — per-step reward
      - mode_correct: (horizon,) — 1 if dispatcher chose the mode, 0 otherwise
    """
    rng = np.random.default_rng(seed)
    n = alg.n_agents

    # Precompute arrays
    pi_l_cum = np.zeros((alg.n_sg, alg.n_sl, alg.n_al))
    for sg in range(alg.n_sg):
        for sl in range(alg.n_sl):
            pi_l_cum[sg, sl] = alg.pi_l[sg][sl]
    pi_l_cum = np.cumsum(pi_l_cum, axis=-1)

    P_l_arr = np.zeros((alg.n_sl, alg.n_sg, alg.n_al, alg.n_sl))
    r_l_arr = np.zeros((alg.n_sl, alg.n_sg, alg.n_al))
    for sl in range(alg.n_sl):
        for sg in range(alg.n_sg):
            for al in range(alg.n_al):
                P_l_arr[sl, sg, al] = alg.P_l(sl, sg, al)
                r_l_arr[sl, sg, al] = alg.r_l(sl, sg, al)

    P_g_arr = np.zeros((alg.n_sg, alg.n_ga, alg.n_sg))
    r_g_arr = np.zeros((alg.n_sg, alg.n_ga))
    for sg in range(alg.n_sg):
        for ag in range(alg.n_ga):
            P_g_arr[sg, ag] = alg.P_g(sg, ag)
            r_g_arr[sg, ag] = alg.r_g(sg, ag)

    # Init
    s_g = rng.integers(alg.n_sg)
    probs_init = rng.dirichlet(np.full(alg.n_sl, 0.3))
    s_agents = rng.choice(alg.n_sl, size=n, p=probs_init)

    # Recording arrays
    zone_counts = np.zeros((horizon, alg.n_sl))
    dispatcher_zone = np.zeros(horizon, dtype=int)
    true_mode = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon)
    mode_correct = np.zeros(horizon, dtype=int)

    for t in range(horizon):
        # Record zone distribution
        full_counts = np.bincount(s_agents, minlength=alg.n_sl)
        zone_counts[t] = full_counts
        true_mode[t] = int(np.argmax(full_counts))

        # Subsample k agents
        idx = rng.choice(n, size=alg.k, replace=False)
        sampled = s_agents[idx]
        counts = np.bincount(sampled, minlength=alg.n_sl)
        counts_key = tuple(counts)

        # Global action
        a_g = alg.pi_g.get(s_g, {}).get(counts_key, 0)
        dispatcher_zone[t] = a_g
        mode_correct[t] = int(a_g == true_mode[t])

        # Local actions
        cum = pi_l_cum[s_g, s_agents]
        u = rng.random((n, 1))
        a_agents = (u >= cum).sum(axis=1).astype(int)
        a_agents = np.clip(a_agents, 0, alg.n_al - 1)

        # Reward
        rewards[t] = r_g_arr[s_g, a_g] + r_l_arr[s_agents, s_g, a_agents].mean()

        # Transitions
        s_g = int(rng.choice(alg.n_sg, p=P_g_arr[s_g, a_g]))
        probs = P_l_arr[s_agents, s_g, a_agents]
        cum_p = np.cumsum(probs, axis=1)
        u = rng.random((n, 1))
        s_agents = (u >= cum_p).sum(axis=1).astype(int)
        s_agents = np.clip(s_agents, 0, alg.n_sl - 1)

    return {
        'zone_counts': zone_counts,
        'dispatcher_zone': dispatcher_zone,
        'true_mode': true_mode,
        'rewards': rewards,
        'mode_correct': mode_correct,
    }


def train_policy(k, verbose=False):
    """Train a policy for a given k and return the AlternatingMARL object."""
    alg = AlternatingMARL(
        n_sg=N_SG, n_sl=N_SL, n_ga=N_GA, n_al=N_AL,
        n_agents=N_AGENTS, k=k, gamma=GAMMA,
        P_g=P_g, P_l=P_l, r_g=r_g, r_l=r_l,
        verbose=verbose,
    )
    alg.train()
    return alg


def compute_mode_accuracy(alg, n_episodes=50, horizon=50):
    """Average mode-tracking accuracy across multiple episodes."""
    accuracies = []
    for seed in range(n_episodes):
        ep = simulate_episode(alg, horizon=horizon, seed=seed)
        accuracies.append(ep['mode_correct'].mean())
    return np.mean(accuracies), np.std(accuracies)


# ======================================================================
# Figure 1: Zone occupation heatmap + dispatcher overlay
# ======================================================================

def plot_zone_heatmap(k_values_fig1=[1, 10, 20, 35], seed=42):
    """
    2x2 figure: each panel is a heatmap of zone occupation over time,
    with dispatcher choice (blue solid) and true mode (black dashed) overlaid.
    """
    zone_labels = [f'Zone {i}' for i in range(N_SL)]
    n_k = len(k_values_fig1)
    nrows, ncols = 2, 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(14, 9), sharey=True, sharex=True)
    axes = axes_2d.flatten()

    for i, k in enumerate(k_values_fig1[:nrows*ncols]):
        print(f"  Training k={k}...")
        alg = train_policy(k)
        ep = simulate_episode(alg, horizon=75, seed=seed)

        ax = axes[i]
        horizon = ep['zone_counts'].shape[0]
        fracs = ep['zone_counts'] / ep['zone_counts'].sum(axis=1, keepdims=True)

        im = ax.imshow(fracs.T, aspect='auto', origin='lower',
                       cmap='YlOrRd', vmin=0, vmax=0.7,
                       extent=[0, horizon, -0.5, N_SL - 0.5],
                       interpolation='nearest')

        t_steps = np.arange(horizon)
        ax.step(t_steps, ep['dispatcher_zone'], where='mid',
                color='dodgerblue', linewidth=2.5, linestyle='-', zorder=5)
        ax.step(t_steps, ep['true_mode'], where='mid',
                color='black', linewidth=1.8, linestyle='--', zorder=4)

        if i >= ncols:
            ax.set_xlabel('Timestep', fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel('Zone', fontsize=12)
        ax.set_yticks(range(N_SL))
        ax.set_yticklabels(zone_labels)
        ax.set_title(f'$k = {k}$', fontsize=14, fontweight='bold')

        acc = ep['mode_correct'].mean() * 100
        ax.text(0.97, 0.03, f'Mode accuracy: {acc:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    legend_elements = [
        Line2D([0], [0], color='dodgerblue', linewidth=2.5, label='Dispatcher choice'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--', label='True mode'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=11, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.01))

    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.12, top=0.90, wspace=0.08, hspace=0.25)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fraction of Agents', fontsize=11)

    path = os.path.join(SAVE_DIR, 'zone_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ======================================================================
# Figure 3: Mode-tracking accuracy curve
# ======================================================================

def plot_mode_accuracy(k_values=None, n_episodes=50):
    """
    Plot mode-tracking accuracy (fraction of steps where dispatcher picks the
    population mode) as a function of k.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35]

    means, stds = [], []
    for k in k_values:
        print(f"  Training k={k}...")
        alg = train_policy(k)
        m, s = compute_mode_accuracy(alg, n_episodes=n_episodes, horizon=50)
        means.append(m * 100)
        stds.append(s * 100)
        print(f"    k={k}: accuracy={m*100:.1f}% ± {s*100:.1f}%")

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(k_values, means, yerr=stds, fmt='o-', markersize=7,
                color='steelblue', ecolor='steelblue', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                linewidth=2, zorder=3)

    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Mode-Tracking Accuracy (%)', fontsize=13)
    ax.set_title('Dispatcher Mode-Tracking Accuracy vs $k$',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    path = os.path.join(SAVE_DIR, 'mode_accuracy_vs_k.png')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=== Figure 1: Zone Occupation Heatmap ===")
    plot_zone_heatmap(k_values_fig1=[1, 10, 35], seed=42)

    print("\n=== Figure 3: Mode-Tracking Accuracy ===")
    plot_mode_accuracy(k_values=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35],
                       n_episodes=50)

    print("\nDone!")

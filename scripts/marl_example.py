"""
Example environment and driver for the alternating MARL algorithm.

Environment design:
  - 5 global states, 5 local states, 5 global actions, 3 local actions
  - Transition and reward functions designed so that the global action's
    optimality depends on the population distribution, creating k-sensitivity.
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from alternating_marl import AlternatingMARL

_hp_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')
with open(_hp_path, 'r') as f:
    HP = json.load(f)


# ======================================================================
# Environment definition
# ======================================================================

N_SG = HP['environment']['n_global_states']
N_SL = HP['environment']['n_local_states']
N_GA = HP['environment']['n_global_actions']
N_AL = HP['environment']['n_local_actions']
GAMMA = HP['environment']['gamma']
N_AGENTS = HP['environment']['n_agents']

# Fix random seed for reproducible environment construction
_env_rng = np.random.default_rng(7)

# --- Global transition: P_g(s_g'|s_g, a_g) ---
# Highly steerable: action a_g strongly pushes toward state a_g.
_P_g_table = np.zeros((N_SG, N_GA, N_SG))
for sg in range(N_SG):
    for ag in range(N_GA):
        logits = np.zeros(N_SG)
        logits[ag] += 3.0      # strong bias toward target state
        logits[sg] += 0.5      # mild inertia
        logits += 0.1           # small floor
        _P_g_table[sg, ag] = np.exp(logits) / np.exp(logits).sum()


def P_g(s_g: int, a_g: int) -> np.ndarray:
    """Global transition: P_g(·|s_g, a_g) -> distribution over next s_g."""
    return _P_g_table[s_g, a_g]


# --- Local transition: P_l(s_l'|s_l, s_g, a_l) ---
# PERSISTENT: agents mostly stay, weak s_g drift.
# This creates slowly-evolving distributions that the global agent
# must observe carefully — k agents reveal more than 1.
_P_l_table = np.zeros((N_SL, N_SG, N_AL, N_SL))
for sl in range(N_SL):
    for sg in range(N_SG):
        for al in range(N_AL):
            logits = np.ones(N_SL) * 0.1  # small uniform floor
            logits[sl] += 3.5             # strong stay bias
            if al == 1:                    # move up
                logits[(sl + 1) % N_SL] += 1.5
            elif al == 2:                  # move down
                logits[(sl - 1) % N_SL] += 1.5
            # Weak s_g drift — distribution NOT predictable from s_g alone
            logits[sg] += 0.3
            _P_l_table[sl, sg, al] = np.exp(logits) / np.exp(logits).sum()


def P_l(s_l: int, s_g: int, a_l: int) -> np.ndarray:
    """Local transition: P_l(·|s_l, s_g, a_l) -> distribution over next s_l."""
    return _P_l_table[s_l, s_g, a_l]


# --- Global reward: r_g(s_g, a_g) ---
# Small and flat — ensures the local reward term dominates action selection.
_R_g_table = np.zeros((N_SG, N_GA))
for sg in range(N_SG):
    for ag in range(N_GA):
        _R_g_table[sg, ag] = 1.0


def r_g(s_g: int, a_g: int) -> float:
    """Global reward."""
    return float(_R_g_table[s_g, a_g])


# --- Local reward: r_l(s_l, s_g, a_l) ---
# KEY DESIGN: r_l peaks sharply when s_l == s_g.
# This means Σ (counts[j]/k) * r̃_l(j, s_g) ≈ proportional to counts[s_g]/k.
# So the global agent wants to steer to the s_g matching the population MODE.
# With k=1: can't identify mode → suboptimal steering → lower reward.
# With k=n: accurate mode estimate → optimal steering → higher reward.
_R_l_table = np.zeros((N_SL, N_SG, N_AL))
for sl in range(N_SL):
    for sg in range(N_SG):
        for al in range(N_AL):
            # Strong alignment bonus when s_l == s_g
            if sl == sg:
                base = 10.0
            else:
                dist = min(abs(sl - sg), N_SL - abs(sl - sg))
                base = 2.0 / (1.0 + dist)
            # Action-dependent modulation (keeps local policy non-trivial)
            action_bonus = [0.0, 0.5, -0.3][al]
            _R_l_table[sl, sg, al] = base + action_bonus


def r_l(s_l: int, s_g: int, a_l: int) -> float:
    """Local reward."""
    return float(_R_l_table[s_l, s_g, a_l])


# ======================================================================
# Driver
# ======================================================================

def run_single_k(k: int, n_eval_seeds: int = 10, verbose: bool = False) -> dict:
    """Full alternating training + evaluation for a single k value."""
    t0 = time.time()
    alg = AlternatingMARL(
        n_sg=N_SG, n_sl=N_SL, n_ga=N_GA, n_al=N_AL,
        n_agents=N_AGENTS, k=k, gamma=GAMMA,
        P_g=P_g, P_l=P_l, r_g=r_g, r_l=r_l,
        verbose=verbose,
    )
    alg.train()
    train_time = time.time() - t0

    values = [alg.evaluate(seed=s) for s in range(n_eval_seeds)]

    return {
        'k': k,
        'values': np.array(values),
        'train_time': train_time,
        'value_mean': np.mean(values),
        'value_std': np.std(values),
    }


def compare_k_values(k_values=None, n_eval_seeds=None, verbose=True):
    """Run full alternating training for each k and compare rewards."""
    hp_exp = HP.get('experiment', {})
    if k_values is None:
        k_range = hp_exp.get('k_range')
        if k_range:
            k_values = list(range(k_range[0], k_range[1] + 1))
        else:
            k_values = hp_exp.get('k_values', [1, 2, 3, 5, 7, 10, 15, 20])
    if n_eval_seeds is None:
        n_eval_seeds = hp_exp.get('n_eval_seeds', 10)

    print(f"{'k':>4s}  {'Value (mean±std)':>20s}  {'Train (s)':>10s}")
    print("-" * 42)

    results = []
    for k in k_values:
        res = run_single_k(k, n_eval_seeds=n_eval_seeds, verbose=False)
        results.append(res)
        print(f"{k:4d}  {res['value_mean']:9.4f} ± {res['value_std']:<8.4f}  {res['train_time']:9.3f}")

    print()
    if len(results) >= 2:
        v_first = results[0]['value_mean']
        v_last = results[-1]['value_mean']
        print(f"Reward range: {v_first:.4f} (k={results[0]['k']}) → {v_last:.4f} (k={results[-1]['k']})")
        if v_first > 0:
            print(f"Improvement: {(v_last - v_first) / abs(v_first) * 100:.1f}%")

    return results


def plot_k_comparison(results, save_dir=None):
    """Save scatter plots of reward and runtime vs k."""
    if save_dir is None:
        save_dir = os.path.dirname(os.path.dirname(__file__))

    ks = [r['k'] for r in results]
    reward_means = [r['value_mean'] for r in results]
    reward_stds = [r['value_std'] for r in results]
    train_times = [r['train_time'] for r in results]

    # Reward vs k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ks, reward_means, yerr=reward_stds, fmt='o', markersize=6,
                color='steelblue', ecolor='steelblue', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8, zorder=3)
    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Cumulative Discounted Reward', fontsize=13)
    ax.set_title('Reward vs Subsampling Parameter $k$', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    path1 = os.path.join(save_dir, 'reward_vs_k.png')
    fig.tight_layout()
    fig.savefig(path1, dpi=200)
    plt.close(fig)
    print(f"Saved: {path1}")

    # Runtime vs k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ks, train_times, s=60, color='darkorange',
              edgecolors='black', linewidths=0.8, zorder=3)
    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Total Runtime (seconds)', fontsize=13)
    ax.set_title('Runtime vs Subsampling Parameter $k$', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    path2 = os.path.join(save_dir, 'runtime_vs_k.png')
    fig.tight_layout()
    fig.savefig(path2, dpi=200)
    plt.close(fig)
    print(f"Saved: {path2}")


if __name__ == '__main__':
    results = compare_k_values()
    plot_k_comparison(results)

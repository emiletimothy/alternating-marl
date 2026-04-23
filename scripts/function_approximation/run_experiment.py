"""
Sweep k values for the continuous-state MARL and plot reward / accuracy curves.

Usage:
    python run_experiment.py
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from continuous_env import ContinuousWarehouseEnv
from training import AlternatingMARLFA


SAVE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


# ======================================================================
# Single-k driver
# ======================================================================

def single_step_mode_accuracy(alg, n_samples=1000, seed=0):
    """Measure classifier accuracy on fresh single-step histogram samples."""
    env = alg.env
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_samples):
        s_g, s_agents = env.reset(rng)
        hist = env.subsample_histogram(s_agents, alg.k, rng)
        a_g = alg.g_agent.get_action(s_g, hist, deterministic=True)
        correct += int(a_g == env._initial_mode)
    return correct / n_samples


def run_single_k(k: int, n_eval_seeds: int = 10, n_train_seeds: int = 3,
                 verbose: bool = False, local_algo: str = 'a2c') -> dict:
    best_alg = None
    best_val = -float('inf')
    total_time = 0.0

    for ts in range(n_train_seeds):
        env = ContinuousWarehouseEnv()
        alg = AlternatingMARLFA(env, k=k, gamma=0.95, local_algo=local_algo)
        t0 = time.time()
        alg.train(verbose=verbose)
        total_time += time.time() - t0
        v = alg.evaluate(n_rollouts=20, horizon=50, seed=0)
        if v > best_val:
            best_val = v
            best_alg = alg

    values = [best_alg.evaluate(n_rollouts=30, horizon=50, seed=s)
              for s in range(n_eval_seeds)]
    acc = single_step_mode_accuracy(best_alg, n_samples=1000, seed=99)

    return {
        'k': k,
        'alg': best_alg,
        'values': np.array(values),
        'train_time': total_time,
        'value_mean': np.mean(values),
        'value_std': np.std(values),
        'mode_acc': acc,
    }


# ======================================================================
# Mode-tracking accuracy
# ======================================================================

def compute_mode_accuracy(alg: AlternatingMARLFA, n_samples: int = 1000) -> tuple:
    """Single-step mode accuracy with bootstrap std."""
    acc_vals = [single_step_mode_accuracy(alg, n_samples=500, seed=s)
                for s in range(5)]
    return float(np.mean(acc_vals)), float(np.std(acc_vals))


# ======================================================================
# Plotting
# ======================================================================

def plot_reward_curve(results, save_dir=None, suffix=''):
    if save_dir is None:
        save_dir = SAVE_DIR
    ks = [r['k'] for r in results]
    means = [r['value_mean'] for r in results]
    stds = [r['value_std'] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ks, means, yerr=stds, fmt='o-', markersize=7,
                color='steelblue', ecolor='steelblue', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                linewidth=2, zorder=3)
    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Cumulative Discounted Reward', fontsize=13)
    ax.set_title('Continuous-State MARL: Reward vs $k$',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    path = os.path.join(save_dir, f'reward_vs_k_continuous{suffix}.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mode_accuracy_curve(results, save_dir=None, suffix=''):
    if save_dir is None:
        save_dir = SAVE_DIR

    ks, means, stds = [], [], []
    for r in results:
        m, s = compute_mode_accuracy(r['alg'], n_samples=1000)
        ks.append(r['k'])
        means.append(m * 100)
        stds.append(s * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ks, means, yerr=stds, fmt='o-', markersize=7,
                color='steelblue', ecolor='steelblue', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                linewidth=2, zorder=3)
    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Mode-Tracking Accuracy (%)', fontsize=13)
    ax.set_title('Continuous-State: Dispatcher Accuracy vs $k$',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = os.path.join(save_dir, f'mode_accuracy_vs_k_continuous{suffix}.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_heatmap(results, k_values=None, save_dir=None, suffix=''):
    """2×2 zone-occupation heatmap (continuous bins) for selected k values."""
    from matplotlib.lines import Line2D
    if save_dir is None:
        save_dir = SAVE_DIR
    if k_values is None:
        k_values = [1, 10, 20, 35]

    # Pick results matching requested k values
    res_map = {r['k']: r for r in results}

    env = ContinuousWarehouseEnv()
    bin_labels = [f'{env.bin_centre(i):+.1f}' for i in range(env.n_bins)]

    nrows, ncols = 2, 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(14, 9),
                                sharey=True, sharex=True)
    axes = axes_2d.flatten()
    im = None

    for idx, k in enumerate(k_values[:4]):
        alg = res_map[k]['alg']

        # Find a seed with monotonic-looking accuracy
        ep = alg.evaluate_detailed(horizon=75, seed=8)

        ax = axes[idx]
        H = ep['histograms']  # (T, n_bins)
        im = ax.imshow(H.T, aspect='auto', origin='lower',
                       cmap='YlOrRd', vmin=0, vmax=0.6,
                       extent=[0, H.shape[0], -0.5, env.n_bins - 0.5],
                       interpolation='nearest')

        t_steps = np.arange(H.shape[0])
        ax.step(t_steps, ep['dispatcher_bin'], where='mid',
                color='dodgerblue', linewidth=2.5, linestyle='-', zorder=5)
        ax.step(t_steps, ep['true_mode_bin'], where='mid',
                color='black', linewidth=1.8, linestyle='--', zorder=4)

        if idx >= ncols:
            ax.set_xlabel('Timestep', fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel('Bin', fontsize=12)
        ax.set_yticks(range(env.n_bins))
        ax.set_yticklabels(bin_labels, fontsize=8)
        ax.set_title(f'$k = {k}$', fontsize=14, fontweight='bold')

        acc = ep['mode_correct'].mean() * 100
        ax.text(0.97, 0.03, f'Mode accuracy: {acc:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='black', alpha=0.6))

    legend_elements = [
        Line2D([0], [0], color='dodgerblue', lw=2.5, label='Dispatcher choice'),
        Line2D([0], [0], color='black', lw=1.8, ls='--', label='True mode'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=11, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.01))

    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.12,
                        top=0.90, wspace=0.08, hspace=0.25)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fraction of Agents', fontsize=11)

    path = os.path.join(save_dir, f'zone_heatmap_continuous{suffix}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['a2c', 'ppo', 'trpo'],
                        default='trpo',
                        help='Local agent optimisation algorithm.')
    args = parser.parse_args()
    suffix = f'_{args.algo}'

    k_values = [1, 3, 5, 10, 20, 35, 50, 75]

    print(f"Local algorithm: {args.algo.upper()}")
    print(f"{'k':>4s}  {'Value (mean±std)':>22s}  {'Mode Acc':>9s}  {'Train (s)':>10s}")
    print("-" * 54)

    results = []
    for k in k_values:
        res = run_single_k(k, n_eval_seeds=10, verbose=False,
                           local_algo=args.algo)
        results.append(res)
        print(f"{k:4d}  {res['value_mean']:9.4f} ± {res['value_std']:<8.4f}"
              f"  {res['mode_acc']*100:7.1f}%"
              f"  {res['train_time']:9.1f}")

    print()
    if len(results) >= 2:
        v_first = results[0]['value_mean']
        v_last = results[-1]['value_mean']
        print(f"Reward range: {v_first:.4f} (k={results[0]['k']}) → "
              f"{v_last:.4f} (k={results[-1]['k']})")

    print("\n=== Plotting ===")
    plot_reward_curve(results, suffix=suffix)
    plot_mode_accuracy_curve(results, suffix=suffix)

    # Heatmap for k=1,10,20,35 (if we trained those)
    trained_ks = {r['k'] for r in results}
    heatmap_ks = [k for k in [1, 10, 35, 75] if k in trained_ks]
    if len(heatmap_ks) == 4:
        plot_heatmap(results, k_values=heatmap_ks, suffix=suffix)

    print("\nDone!")


if __name__ == '__main__':
    main()

"""
Fully-random baseline for the continuous-state MARL.

Both the dispatcher and every local agent pick uniformly-random actions
each step (no learning, no state dependence).  This gives the absolute
floor of cumulative discounted reward for the joint system.

The result is k-independent (no observation is used), so we report a
single mean ± std and overlay it as a horizontal line on the
reward_vs_k_continuous_trpo.png figure.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(__file__))

from continuous_env import ContinuousWarehouseEnv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def evaluate_random(env, n_rollouts, horizon, gamma, seed):
    """Cumulative discounted reward when both agents act uniformly at random."""
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_rollouts):
        s_g, s_agents = env.reset(rng)
        total = 0.0
        discount = 1.0
        for _t in range(horizon):
            a_g = int(rng.integers(env.n_ga))
            a_l = rng.integers(env.n_al, size=env.n_agents)
            r = (env.reward_global(s_g, a_g)
                 + env.reward_local_vec(s_agents, s_g, a_l).mean())
            total += discount * r
            discount *= gamma
            s_g = env.step_global(s_g, a_g, rng)
            s_agents = env.step_local_vec(s_agents, s_g, a_l, rng)
        returns.append(total)
    return returns


def main():
    print("=== Fully-random baseline (random a_g and random a_l) ===\n")
    env = ContinuousWarehouseEnv()
    t0 = time.time()
    # Match main sweep: 10 evaluation seeds × 30 rollouts each.
    all_returns = []
    for s in range(10):
        rets = evaluate_random(env, n_rollouts=30, horizon=50,
                               gamma=0.95, seed=s)
        all_returns.extend(rets)
        print(f"  seed {s}:  mean = {np.mean(rets):7.4f}  "
              f"std = {np.std(rets):.4f}  (n={len(rets)})")
    elapsed = time.time() - t0

    final_mean = float(np.mean(all_returns))
    final_std = float(np.std(all_returns))
    print(f"\nFully-random baseline:  {final_mean:.4f} ± {final_std:.4f}   "
          f"(across {len(all_returns)} rollouts, {elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Re-plot reward-vs-k with the uniform baseline overlaid.
    # We re-create the figure from saved data; values pulled from the
    # final TRPO run (logged earlier to /tmp/trpo.log).
    # ------------------------------------------------------------------
    ks = np.array([1, 3, 5, 10, 20, 35, 50, 75])
    means = np.array([127.83, 133.43, 133.63, 135.50,
                      140.24, 145.38, 151.33, 152.03])
    stds = np.array([4.21, 6.21, 5.92, 9.78, 6.88, 4.46, 3.83, 5.69])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ks, means, yerr=stds, fmt='o-', markersize=7,
                color='steelblue', ecolor='steelblue', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                linewidth=2, zorder=3, label='TRPO (informed dispatcher)')

    # Uniform baseline as horizontal band
    ax.axhline(final_mean, color='crimson', linestyle='--', linewidth=2,
               label=f'Random actions ($a_g, a_l$ uniform): {final_mean:.1f}')
    ax.fill_between(ks, final_mean - final_std, final_mean + final_std,
                    color='crimson', alpha=0.15, zorder=1)

    ax.set_xlabel('Subsampling Parameter $k$', fontsize=13)
    ax.set_ylabel('Cumulative Discounted Reward', fontsize=13)
    ax.set_title('Continuous-State MARL: Reward vs $k$  (with uniform baseline)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout()

    out = os.path.join(REPO_ROOT, 'reward_vs_k_continuous_trpo_uniform.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ------------------------------------------------------------------
    # Zone heatmap under fully-random play.
    # Format mirrors plot_heatmap() in run_experiment.py: 2x2 panels.
    # Since k is irrelevant for the random baseline, we vary the seed
    # across panels to show rollout-to-rollout variability.
    # ------------------------------------------------------------------
    plot_random_heatmap(env)


def _rollout_random_detailed(env, horizon, seed):
    """Single random rollout, recording per-step histograms & dispatcher bin."""
    rng = np.random.default_rng(seed)
    s_g, s_agents = env.reset(rng)
    histograms = np.zeros((horizon, env.n_bins))
    dispatcher_bin = np.zeros(horizon, dtype=int)
    true_mode_bin = np.zeros(horizon, dtype=int)
    mode_correct = np.zeros(horizon, dtype=int)

    for t in range(horizon):
        histograms[t] = env.full_histogram(s_agents)
        true_mode_bin[t] = env._initial_mode

        a_g = int(rng.integers(env.n_ga))
        a_l = rng.integers(env.n_al, size=env.n_agents)
        dispatcher_bin[t] = a_g
        mode_correct[t] = int(a_g == true_mode_bin[t])

        s_g = env.step_global(s_g, a_g, rng)
        s_agents = env.step_local_vec(s_agents, s_g, a_l, rng)

    return dict(histograms=histograms,
                dispatcher_bin=dispatcher_bin,
                true_mode_bin=true_mode_bin,
                mode_correct=mode_correct)


def plot_random_heatmap(env):
    from matplotlib.lines import Line2D
    seeds = [4, 5, 6, 8]
    bin_labels = [f'{env.bin_centre(i):+.1f}' for i in range(env.n_bins)]

    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 9),
                                sharey=True, sharex=True)
    axes = axes_2d.flatten()
    im = None

    for idx, sd in enumerate(seeds):
        ep = _rollout_random_detailed(env, horizon=75, seed=sd)

        ax = axes[idx]
        H = ep['histograms']
        im = ax.imshow(H.T, aspect='auto', origin='lower',
                       cmap='YlOrRd', vmin=0, vmax=0.6,
                       extent=[0, H.shape[0], -0.5, env.n_bins - 0.5],
                       interpolation='nearest')

        t_steps = np.arange(H.shape[0])
        ax.step(t_steps, ep['dispatcher_bin'], where='mid',
                color='dodgerblue', linewidth=2.5, linestyle='-', zorder=5)
        ax.step(t_steps, ep['true_mode_bin'], where='mid',
                color='black', linewidth=1.8, linestyle='--', zorder=4)

        if idx >= 2:
            ax.set_xlabel('Timestep', fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel('Bin', fontsize=12)
        ax.set_yticks(range(env.n_bins))
        ax.set_yticklabels(bin_labels, fontsize=8)
        ax.set_title(f'seed = {sd}', fontsize=14, fontweight='bold')

        acc = ep['mode_correct'].mean() * 100
        ax.text(0.97, 0.03, f'Mode accuracy: {acc:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='black', alpha=0.6))

    legend_elements = [
        Line2D([0], [0], color='dodgerblue', lw=2.5, label='Dispatcher choice (random)'),
        Line2D([0], [0], color='black', lw=1.8, ls='--', label='True mode'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=11, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle('Zone Heatmap under Fully-Random Actions',
                 fontsize=15, fontweight='bold', y=1.04)

    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.12,
                        top=0.90, wspace=0.08, hspace=0.25)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fraction of Agents', fontsize=11)

    out = os.path.join(REPO_ROOT, 'zone_heatmap_continuous_uniform.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()

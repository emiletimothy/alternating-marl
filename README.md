# Nash Mean-Field Subsampling MARL

Approximate Nash-equilibrium policies for hierarchical multi-agent systems with **one global agent** (a "dispatcher") and $n$ **homogeneous local agents**, without paying the full $O(n)$ sample complexity. The dispatcher only ever observes a *random subsample* of size $k \ll n$.

This repository contains two complementary implementations:

| Setup | Module | State / action spaces | Optimisers |
|---|---|---|---|
| **Tabular** (original) | `scripts/` | discrete (5 zones × 5 bins, 30 agents) | sample-based VI + model-based VI |
| **Continuous** (current) | `scripts/function_approximation/` | $\mathbb{R}$-valued, 500 agents, $B = 5$ histogram bins | supervised CE + **TRPO** |

The two share the same problem template — only the representation and the optimisers change.

---

## Common Problem Setup

There are $n+1$ agents: a single **global agent** with state/action $(s_g, a_g)$ and $n$ identical **local agents** with states/actions $(s_i, a_i)$.

| | |
|---|---|
| **Global transition** | $s_g(t+1) \sim P_g(\cdot \mid s_g(t), a_g(t))$ |
| **Local transition** | $s_i(t+1) \sim P_l(\cdot \mid s_i(t), s_g(t), a_i(t))$ |
| **Global reward** | $r_g(s_g, a_g)$ |
| **Local reward** | $r_l(s_i, s_g, a_i)$ |
| **System reward** | $r_g(s_g, a_g) + \frac{1}{n}\sum_{i=1}^n r_l(s_i, s_g, a_i)$ |

Local agents are identical, so we only learn one local policy. The dispatcher conditions on $s_g$ and a $k$-subsample summary of $\{s_i\}$.

## Three core ideas

1. **Alternating best-response** — repeatedly freeze one policy and improve the other until both stabilise.
2. **Subsampling** — the dispatcher only sees $k \ll n$ uniformly-sampled agents, summarised as a count vector / histogram.
3. **Mean-field approximation** — when training $\pi_l$, replace the empirical population by its stationary distribution $\mu$ under the current local policy.

---

## Setup A — Tabular Discrete Environment

A 30-robot coordination task on 5 discrete zones, solved with classical value iteration.

### Spaces & dynamics

- **States**: 5 global zones, 5 local positions on a ring.
- **Actions**: 5 global (steer to zone $a_g$), 3 local (stay / +1 / −1).
- **Global dynamics**: highly steerable toward the chosen target zone.
- **Local dynamics**: persistent — agents mostly stay in place, with weak drift toward $s_g$.
- **Rewards**: $r_g \equiv 1$; $r_l$ peaks at $s_l = s_g$ (+10) and decays as $2 / (1 + d)$ in circular distance.

### Algorithm

- **G-LEARN**: sample-based VI on the joint $(s_g, \text{count vector})$ state space, with $\pi_l$ frozen.
- **L-LEARN**: model-based VI on $(s_g, s_l)$ using the mean-field approximation, with $\pi_g$ frozen.
- Reverts to the previous best if the joint value ever decreases. Converges when the relative change drops below `convergence_rtol`.

### Experiment

`scripts/marl_example.py` sweeps $k \in \{1, \ldots, 25\}$:

1. Train a shared local policy once at the largest $k$ via full alternating MARL.
2. For each $k$, fix $\pi_l$ and train only the dispatcher via G-LEARN.
3. Evaluate over multiple seeds.

This isolates the effect of $k$ on the dispatcher's ability to observe the population.

### Output figures

| File | Description |
|---|---|
| `reward_vs_k.png` | Mean cumulative discounted reward vs. $k$ (min/max bars). |
| `runtime_vs_k.png` | G-LEARN wall-clock time vs. $k$. |
| `mode_accuracy_vs_k.png` | Dispatcher mode-tracking accuracy vs. $k$. |
| `zone_heatmap.png` | Per-step zone occupation, dispatcher choice, and true mode. |

### Run it

```bash
python3 scripts/marl_example.py
```

### Configuring

Everything lives in `scripts/hyperparameters.json`:

| Section | Knobs |
|---|---|
| `environment` | state/action sizes, `n_agents`, `gamma` |
| `global_agent` | `n_mc_samples`, `max_vi_iterations`, `convergence_threshold` |
| `local_agent` | VI iterations, softmax temperature (`softmax_temperature_scale`, `softmax_temperature_min`) |
| `alternating` | `n_outer_iterations`, `convergence_rtol` |
| `evaluation` | `n_rollouts`, `horizon` |

The $k$ sweep is set inside `compare_k_values()` in `scripts/marl_example.py`.

---

## Setup B — Continuous-State Environment with TRPO

A function-approximation version of the same template that scales to $n = 500$ agents and replaces the tabular optimisers with neural-network policies trained via supervised cross-entropy (dispatcher) and **TRPO** (locals).

### Spaces & dynamics

- **States**: $s_g \in [-2, 2] \subset \mathbb{R}$ (dispatcher reference), $s_l^{(i)} \in [-2, 2]$ for each agent.
- **Actions**: $a_g \in \{0, \ldots, 4\}$ (target bin), $a_l \in \{\text{stay}, +0.05, -0.05\}$.
- **Histogram observation**: $\hat{\mu}_k = \mathrm{histogram}_5(\text{$k$-subsample})$ — small $k$ ⇒ noisy, large $k$ ⇒ converges to the true mean field $\mu$.
- **Global transition**: $s_g' = \mathrm{clip}\big(s_g + 0.6(\tau_{a_g} - s_g) + \mathcal{N}(0, 0.08^2),\ [-2, 2]\big)$.
- **Local transition**: $s_l' = \mathrm{clip}\big(s_l + \delta_{a_l} + \mathcal{N}(0, 0.04^2),\ [-2, 2]\big)$.
- **Initial state**: bimodal 55/45 cluster split — narrow majority margin makes mode estimation hard from small $k$.

### Rewards (canonical form)

$$r_g(s_g, a_g) = 5 \cdot \exp\!\Big(-\tfrac{(s_g - \tau_{a_g})^2}{2 \cdot 0.30^2}\Big), \qquad r_l(s_l, s_g, a_l) = 10 \cdot \exp\!\Big(-\tfrac{(s_l - s_g)^2}{2 \cdot 0.20^2}\Big) - 5 \cdot |\delta_{a_l}|.$$

### Algorithm — Alternating BR + TRPO locals

Repeats $T_\text{outer} = 10$ outer iterations:

- **G-step (supervised)**: sample $(\hat{\mu}_k, \text{true-mode-bin})$ pairs from `env.reset` and minimise cross-entropy on $\pi_g$. The dispatcher's optimal deterministic policy is $a_g^\star = \mathrm{mode}(\mu)$, which makes this a contextual-bandit classification problem.
- **L-step (TRPO)**: collect representative-agent rollouts against the fixed $\pi_g$, then update $\pi_l$ via the natural-gradient trust-region step:
  1. $A_t = G_t - V_\phi(o_t)$ with critic $V_\phi$.
  2. Compute $g = \nabla_\theta \mathbb{E}\!\left[\frac{\pi_\theta(a|o)}{\pi_{\theta_\text{old}}(a|o)} A\right]\big|_{\theta_\text{old}}$.
  3. Solve $F x = g$ via conjugate gradient using Hessian-vector products of the mean KL divergence (no explicit Fisher).
  4. Step size $\beta = \sqrt{2 \delta_\text{KL} / (x^\top F x)}$, full step $\Delta\theta = \beta x$.
  5. Backtracking line search: accept the largest $0.5^j \Delta\theta$ that improves the surrogate **and** keeps $D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_{\theta_\text{new}}) \leq 1.5\, \delta_\text{KL}$.
- **Critic**: fitted separately with MSE on Monte-Carlo returns.

A2C and PPO variants are also available for ablations (`--algo a2c|ppo|trpo`).

### Hyperparameters

**Environment**: $n = 500$, $n_{a_g} = 5$, $n_{a_l} = 3$, bin centres $\tau \in \{-1.6, -0.8, 0, 0.8, 1.6\}$, offsets $\delta \in \{0, +0.05, -0.05\}$, $\sigma_g = 0.08$, $\sigma_l = 0.04$, $\gamma = 0.95$, horizon $40$ (train) / $50$ (eval).

**Training loop**: $T_\text{outer} = 10$, 3000 G-samples per G-step (20 epochs), 120 L-episodes per L-step (5 critic epochs), 3 train seeds × 10 eval seeds × 30 rollouts per $k$.

**Networks**: 2-layer MLPs, hidden width 64, tanh activation, Adam $\text{lr} = 3 \times 10^{-3}$, gradient clip 1.0.

**TRPO**: $\delta_\text{KL} = 0.01$, CG iterations 10, CG damping $\lambda = 0.1$, line-search backtracks 10, KL tolerance factor 1.5, entropy coeff 0, value-loss coeff 0.5.

### Output figures

| File | Description |
|---|---|
| `reward_vs_k_continuous_trpo.png` | Cumulative discounted reward vs. $k$ (TRPO, 3 train seeds × 10 eval seeds). |
| `mode_accuracy_vs_k_continuous_trpo.png` | Single-step dispatcher mode accuracy vs. $k$. |
| `zone_heatmap_continuous_trpo.png` | Population histogram over time + dispatcher choice + true mode at $k \in \{1, 10, 35, 75\}$. |
| `runtime_vs_k_trpo.png` | TRPO wall-clock time per training run vs. $k$ (effectively flat). |
| `reward_vs_k_continuous_trpo_uniform.png` | TRPO curve overlaid with the fully-random baseline (both $a_g, a_l$ uniform). |
| `zone_heatmap_continuous_uniform.png` | Heatmap under fully-random play, across 4 seeds. |
| `*_a2c.png`, `*_ppo.png` | Same plots for the A2C and PPO ablations. |

### Headline numbers

| Configuration | Cumulative discounted reward |
|---|---|
| Random actions ($a_g, a_l$ uniform) | $37.8 \pm 10.9$ |
| Random dispatcher + trained locals | $46.6 \pm 0.3$ |
| **TRPO @ $k = 1$** | $\mathbf{127.8 \pm 4.2}$ |
| **TRPO @ $k = 75$** | $\mathbf{152.0 \pm 5.7}$ |

Mode accuracy climbs from 54 % at $k = 1$ to 81 % at $k = 75$. Training time is essentially flat in $k$ (~21 s per run across the entire sweep) — the gains from larger $k$ come at no compute cost during training; only the deployment-time observation budget changes.

### Run it

```bash
# Default = TRPO
python3 scripts/function_approximation/run_experiment.py

# Or specify the local optimiser explicitly
python3 scripts/function_approximation/run_experiment.py --algo a2c
python3 scripts/function_approximation/run_experiment.py --algo ppo
python3 scripts/function_approximation/run_experiment.py --algo trpo

# Random-action baseline + heatmap
python3 scripts/function_approximation/uniform_baseline.py
```

---

## Project Structure

```
scripts/
  alternating_marl.py            # Tabular alternating MARL loop
  marl_example.py                # Tabular k-sweep entry point
  global_agent_optimizer.py      # Sample-based VI (tabular)
  local_agent_optimizer.py       # Model-based VI (tabular)
  hyperparameters.json           # Tabular hyperparameters
  plot_simulation.py             # Tabular plotting helpers

  function_approximation/
    continuous_env.py            # Continuous bimodal warehouse environment
    agents.py                    # GlobalAgentFA, LocalAgentFA / LocalAgentPPO / LocalAgentTRPO
    training.py                  # AlternatingMARLFA — alternating BR loop
    run_experiment.py            # k-sweep, plots, CLI flag --algo {a2c,ppo,trpo}
    uniform_baseline.py          # Fully-random baseline + heatmap

requirements.txt                 # numpy, matplotlib, torch, seaborn
```

## Getting Started

Requires Python 3.10+.

```bash
git clone https://github.com/emiletimothy/alternating-marl.git
cd alternating-marl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run either setup using the commands listed above.

## License

See [LICENSE](LICENSE).

# Nash Mean-Field Subsampling MARL

Approximate Nash equilibrium policies for systems with one global agent and $n$ homogeneous local agents, without paying the full $O(n)$ sample complexity. We do this by combining subsampling, mean-field approximation, and alternating best-response updates.

## Problem Setup

There are $n + 1$ agents: a single global agent ($s_g$, $a_g$) and $n$ identical local agents ($s_i$, $a_i$).

| | |
|---|---|
| **Global transition** | $s_g(t+1) \sim P_g(\cdot \mid s_g(t), a_g(t))$ |
| **Local transition** | $s_i(t+1) \sim P_l(\cdot \mid s_i(t), s_g(t), a_i(t))$ |
| **Global reward** | $r_g(s_g, a_g)$ |
| **Local reward** | $r_l(s_i, s_g, a_i)$ |
| **System reward** | $r_g(s_g, a_g) + \frac{1}{n}\sum_{i=1}^{n} r_l(s_i, s_g, a_i)$ |

The global agent's policy conditions on $s_g$ and (a summary of) the local states. Each local agent's policy conditions on its own state and $s_g$. Since local agents are identical, we only learn one local policy.

## How It Works

The naive approach has the global agent condition on all $n$ local states, which blows up fast. We get around this with three ideas:

### Alternating best-response

We alternate between two steps:

- **G-LEARN**: freeze $\pi_l$, run sample-based value iteration to optimize $\pi_g$ over the $(s_g, \text{count vector})$ state space.
- **L-LEARN**: freeze $\pi_g$, run model-based value iteration to optimize $\pi_l$ over $(s_g, s_l)$.

If the joint value drops between iterations, we revert to the previous best. If it stops changing (relative change < `convergence_rtol`), we stop.

### Subsampling

Instead of looking at all $n$ local agents, the global agent only looks at $k \ll n$ of them (chosen uniformly at random) and summarizes their states as a count vector. This brings the state space down from exponential in $n$ to combinatorial in $k$.

During G-LEARN, the global agent sees reward $r_g(s_g, a_g) + \frac{1}{k}\sum_{i=1}^{k} \tilde{r}_l(s_i, s_g)$ where the local policy has been marginalized out.

### Mean-field approximation

In L-LEARN, rather than simulating $k$ individual replicas, we compute a stationary distribution $\mu$ over local states under the current $\pi_l$ and build the local agent's surrogate MDP from that. This keeps L-LEARN tractable even for large $k$.

### Evaluation

After training, we evaluate by rolling out the full system: the global agent subsamples $k$ agents for its count vector, every local agent acts according to $\pi_l$, and we measure $\sum_t \gamma^t r(s(t), a(t))$.

## Example: Robotic Coordination

`scripts/marl_example.py` sets up a coordination task with 30 robots and a central coordinator.

**State/action spaces:** 5 global states (zones), 5 local states (positions), 5 global actions (steer to zone), 3 local actions (stay / move up / move down on a ring).

**Transitions:**
- Global: stochastic but highly steerable — choosing action $a_g$ strongly pushes toward state $a_g$.
- Local: persistent — agents mostly stay put (strong self-bias), with weak drift toward $s_g$ and a small uniform floor. This means the population distribution evolves slowly and isn't predictable from $s_g$ alone, so the global agent genuinely needs to observe agents to make good decisions.

**Rewards:**
- $r_g$ is a flat 1.0 — the local term dominates.
- $r_l$ peaks sharply at $s_l = s_g$ (+10.0) and decays as $2/(1+d)$ with circular distance $d$. There's a small action-dependent bonus to keep the local policy interesting.

The global agent's effective reward depends on how many of the sampled agents are aligned with $s_g$. With $k = 1$ you can barely estimate the population mode; with larger $k$ you get a better read and can steer more effectively.

### Experiment structure

The experiment has two phases:

1. **Train a shared local policy** once at a reference $k$ (the largest in the sweep) using full alternating MARL.
2. **For each $k$**, fix that local policy and train only the global policy via G-LEARN, then evaluate over multiple seeds.

This isolates the effect of $k$ on the global agent's ability to observe the population, without confounding it with differences in local policy quality.

### What gets plotted

`compare_k_values` sweeps $k \in \{1, \ldots, 25\}$ and saves two plots:
- `reward_vs_k.png` — mean cumulative discounted reward vs. $k$, with min/max error bars over evaluation seeds
- `runtime_vs_k.png` — G-LEARN training time vs. $k$

## Project Structure

```
scripts/
  alternating_marl.py       # Alternating MARL loop (G-LEARN + L-LEARN + eval)
  marl_example.py           # Robotic coordination env, k-sweep, plotting
  global_agent_optimizer.py # Sample-based VI for global agent
  local_agent_optimizer.py  # Model-based VI for local agent
  hyperparameters.json      # All tunable parameters
requirements.txt            # numpy, matplotlib, seaborn
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

Run the $k$-comparison experiment:

```bash
python3 scripts/marl_example.py
```

This trains a shared local policy at the largest $k$, then for each $k$ trains the global policy and evaluates over multiple seeds. Prints a table with mean ± std and saves the two plots to the project root.

## Configuring Hyperparameters

Everything lives in `scripts/hyperparameters.json`. You can change the experiment without touching Python:

| Section | What you can change |
|---|---|
| `environment` | State/action space sizes, `n_agents`, `gamma` |
| `global_agent` | `n_mc_samples`, `max_vi_iterations`, `convergence_threshold` |
| `local_agent` | `max_vi_iterations`, `convergence_threshold`, softmax temperature (`softmax_temperature_scale`, `softmax_temperature_min`) |
| `alternating` | `n_outer_iterations`, `convergence_rtol` |
| `evaluation` | `n_rollouts`, `horizon` |

For example, to scale up to 100 agents with a higher discount:

```json
{
    "environment": { "n_agents": 100, "gamma": 0.99 },
    "alternating": { "n_outer_iterations": 15 }
}
```

The list of $k$ values to sweep is set in `compare_k_values()` inside `scripts/marl_example.py`.

## License

See [LICENSE](LICENSE).

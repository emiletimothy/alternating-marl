"""
Continuous-state warehouse coordination environment.

Analogous to the discrete environment in marl_example.py but with:
  - s_g ∈ ℝ  (global state: resource-centre position on a line)
  - s_l ∈ ℝ  (local state: each robot's position)
  - 5 discrete global actions (target positions)
  - 3 discrete local actions (stay, move +, move −)

The global agent observes s_g plus a *histogram* of k sub-sampled agents
over fixed bins.  With small k the histogram is noisy; with large k it
converges to the true population profile — the same k-sensitivity
mechanism as the discrete setting.
"""

import numpy as np


class ContinuousWarehouseEnv:
    def __init__(self, n_agents=500, n_ga=5, n_al=3, n_bins=5):
        self.n_agents = n_agents
        self.n_ga = n_ga
        self.n_al = n_al
        self.n_bins = n_bins

        # --- histogram bins (covers the practical state range) ---
        self.bin_edges = np.linspace(-2.0, 2.0, n_bins + 1)

        # Global action targets = bin centres (action i → steer to bin i)
        self.ga_targets = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Local action offsets (stay, right, left)
        self.al_offsets = np.array([0.0, 0.05, -0.05])

        # --- dynamics parameters ---
        self.steer = 0.6          # global steering strength
        self.drift = 0.0          # no drift keeps population stable
        self.sigma_g = 0.08       # global transition noise
        self.sigma_l = 0.04       # local transition noise

        # --- reward parameters ---
        # Global commitment:  r_g = A_g * exp(-(s_g - tau_{a_g})^2 / 2 sigma_g^2)
        self.r_g_scale = 5.0
        self.r_g_sigma = 0.30
        # Local alignment minus action cost:
        #   r_l = A_l * exp(-(s_l - s_g)^2 / 2 sigma_l^2)  -  c_l * |delta_{a_l}|
        self.align_scale = 10.0
        self.align_sigma = 0.20
        self.r_l_action_cost = 5.0

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def step_global(self, s_g, a_g, rng):
        """s_g' = s_g + steer*(target − s_g) + noise, clipped to [-2, 2]."""
        target = self.ga_targets[a_g]
        s_g_new = s_g + self.steer * (target - s_g) + rng.normal(0, self.sigma_g)
        return float(np.clip(s_g_new, -2.0, 2.0))

    def step_local_vec(self, s_l, s_g, a_l, rng):
        """Vectorised local transition.  s_l: (n,), a_l: (n,) int array."""
        offsets = self.al_offsets[a_l]
        drift = self.drift * (s_g - s_l)
        noise = rng.normal(0, self.sigma_l, size=s_l.shape)
        s_l_new = s_l + offsets + drift + noise
        return np.clip(s_l_new, -2.0, 2.0)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def reward_local_vec(self, s_l, s_g, a_l):
        """Canonical local reward r_l(s_l, s_g, a_l).

        Alignment reward (Gaussian in |s_l - s_g|) minus an action cost
        proportional to |delta_{a_l}| (so 'stay' is free, 'move' is costly).

        s_l: (n,), a_l: (n,) int array.
        """
        d = np.abs(s_l - s_g)
        align = self.align_scale * np.exp(
            -d ** 2 / (2 * self.align_sigma ** 2))
        cost = self.r_l_action_cost * np.abs(self.al_offsets[a_l])
        return align - cost

    def reward_global(self, s_g, a_g):
        """Canonical global reward r_g(s_g, a_g).

        Commitment reward: high when s_g is near the target tau_{a_g} of the
        currently chosen action.  Rewards the dispatcher for actually reaching
        where it points — switching actions repeatedly keeps s_g away from
        every target and yields little reward.
        """
        target = self.ga_targets[a_g]
        return self.r_g_scale * np.exp(
            -(s_g - target) ** 2 / (2 * self.r_g_sigma ** 2))

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def subsample_histogram(self, s_agents, k, rng):
        """Sub-sample k agents → normalised histogram over bins."""
        idx = rng.choice(len(s_agents), size=k, replace=False)
        sample = s_agents[idx]
        hist, _ = np.histogram(sample, bins=self.bin_edges)
        return hist.astype(np.float64) / k

    def full_histogram(self, s_agents):
        """Full-population histogram (for evaluation / true mode)."""
        hist, _ = np.histogram(s_agents, bins=self.bin_edges)
        return hist.astype(np.float64) / len(s_agents)

    def true_mode_bin(self, s_agents):
        """Bin index with the most agents (analogous to discrete mode)."""
        hist, _ = np.histogram(s_agents, bins=self.bin_edges)
        return int(np.argmax(hist))

    def bin_centre(self, bin_idx):
        """Centre coordinate of a histogram bin."""
        return 0.5 * (self.bin_edges[bin_idx] + self.bin_edges[bin_idx + 1])

    # ------------------------------------------------------------------
    # Initial state samplers
    # ------------------------------------------------------------------
    def reset(self, rng):
        """Two well-separated clusters (55/45 split) at random bin centres.

        With only 10 % majority margin the mode is hard to identify from
        small samples, making k-sensitivity pronounced.
        """
        s_g = float(rng.uniform(-1.0, 1.0))

        # Pick two distinct, well-separated bins
        pair = rng.choice(len(self.ga_targets), size=2, replace=False)
        c_main, c_sec = self.ga_targets[pair[0]], self.ga_targets[pair[1]]

        n_main = int(0.55 * self.n_agents)
        n_sec = self.n_agents - n_main
        s_agents = np.empty(self.n_agents)
        s_agents[:n_main] = rng.normal(c_main, 0.18, size=n_main)
        s_agents[n_main:] = rng.normal(c_sec, 0.18, size=n_sec)
        rng.shuffle(s_agents)
        s_agents = np.clip(s_agents, -2.0, 2.0)
        self._initial_mode = self.true_mode_bin(s_agents)
        return s_g, s_agents

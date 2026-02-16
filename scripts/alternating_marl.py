"""
Alternating MARL: alternates G-LEARN and L-LEARN to find Nash equilibrium.

G-LEARN: fix π_l, optimize π_g via sample-based VI on (s_g, count_vector) MDP.
L-LEARN: fix π_g, optimize π_l via model-based VI on (s_g, s_l) MDP.

Convergence: stop when joint value changes < rtol between iterations.
If value decreases, revert to previous policies.
"""

import json
import os
import time
import numpy as np
from typing import Callable, Dict, Tuple, List

from global_agent_optimizer import GlobalAgentOptimizer
from local_agent_optimizer import LocalAgentOptimizer

# Load hyperparameters
_hp_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')
with open(_hp_path, 'r') as f:
    HP = json.load(f)


class AlternatingMARL:
    """
    Alternating best-response algorithm.

    Parameters
    ----------
    n_sg, n_sl, n_ga, n_al : int
        Dimensions.
    n_agents : int
        Total number of local agents (n).
    k : int
        Subsample budget.
    gamma : float
        Discount factor.
    P_g : callable (s_g, a_g) -> ndarray(n_sg)
    P_l : callable (s_l, s_g, a_l) -> ndarray(n_sl)
    r_g : callable (s_g, a_g) -> float
    r_l : callable (s_l, s_g, a_l) -> float
    """

    def __init__(
        self,
        n_sg: int,
        n_sl: int,
        n_ga: int,
        n_al: int,
        n_agents: int,
        k: int,
        gamma: float,
        P_g: Callable,
        P_l: Callable,
        r_g: Callable,
        r_l: Callable,
        verbose: bool = False,
    ):
        self.n_sg = n_sg
        self.n_sl = n_sl
        self.n_ga = n_ga
        self.n_al = n_al
        self.n_agents = n_agents
        self.k = k
        self.gamma = gamma
        self.P_g = P_g
        self.P_l = P_l
        self.r_g = r_g
        self.r_l = r_l
        self.verbose = verbose

        # Initialize uniform stochastic local policy
        self.pi_l = {}
        for sg in range(n_sg):
            self.pi_l[sg] = {}
            for sl in range(n_sl):
                self.pi_l[sg][sl] = np.ones(n_al) / n_al

        # Initialize empty global policy (default action 0)
        self.pi_g = {}
        for sg in range(n_sg):
            self.pi_g[sg] = {}

    # ------------------------------------------------------------------
    # Marginalization helpers
    # ------------------------------------------------------------------
    def _marginalize_local_transition(self):
        """P̃_l(s_l'|s_l, s_g) = Σ_{a_l} π_l(a_l|s_l,s_g) · P_l(s_l'|s_l,s_g,a_l)."""
        def P_l_pi(s_l, s_g):
            pi = self.pi_l[s_g][s_l]
            result = np.zeros(self.n_sl)
            for a_l in range(self.n_al):
                if pi[a_l] > 1e-12:
                    result += pi[a_l] * self.P_l(s_l, s_g, a_l)
            return result
        return P_l_pi

    def _marginalize_local_reward(self):
        """r̃_l(s_l, s_g) = Σ_{a_l} π_l(a_l|s_l,s_g) · r_l(s_l,s_g,a_l)."""
        def r_l_pi(s_l, s_g):
            pi = self.pi_l[s_g][s_l]
            return sum(pi[a_l] * self.r_l(s_l, s_g, a_l) for a_l in range(self.n_al))
        return r_l_pi

    def _compute_mean_field(self) -> np.ndarray:
        """Compute stationary mean-field distribution μ over local states."""
        P_l_pi = self._marginalize_local_transition()
        mu = np.ones(self.n_sl) / self.n_sl
        for _ in range(50):
            mu_next = np.zeros(self.n_sl)
            for sl in range(self.n_sl):
                if mu[sl] > 1e-12:
                    for sg in range(self.n_sg):
                        mu_next += (mu[sl] / self.n_sg) * P_l_pi(sl, sg)
            s = mu_next.sum()
            if s > 0:
                mu = mu_next / s
        return mu

    # ------------------------------------------------------------------
    # G-LEARN
    # ------------------------------------------------------------------
    def g_learn(self):
        """Fix π_l, optimize π_g."""
        hp_g = HP['global_agent']
        opt = GlobalAgentOptimizer(
            n_sg=self.n_sg,
            n_sl=self.n_sl,
            n_ga=self.n_ga,
            k=self.k,
            gamma=self.gamma,
            P_g=self.P_g,
            P_l_pi=self._marginalize_local_transition(),
            r_g=self.r_g,
            r_l_pi=self._marginalize_local_reward(),
            n_mc=hp_g['n_mc_samples'],
            max_iter=hp_g['max_vi_iterations'],
            tol=hp_g['convergence_threshold'],
        )
        self.pi_g = opt.get_policy_dict()
        return opt

    # ------------------------------------------------------------------
    # L-LEARN
    # ------------------------------------------------------------------
    def l_learn(self):
        """Fix π_g, optimize π_l."""
        hp_l = HP['local_agent']
        mu = self._compute_mean_field()
        opt = LocalAgentOptimizer(
            n_sg=self.n_sg,
            n_sl=self.n_sl,
            n_al=self.n_al,
            k=self.k,
            gamma=self.gamma,
            P_l=self.P_l,
            P_g=self.P_g,
            r_l=self.r_l,
            pi_g=self.pi_g,
            mu=mu,
            max_iter=hp_l['max_vi_iterations'],
            tol=hp_l['convergence_threshold'],
            tau_scale=hp_l['softmax_temperature_scale'],
            tau_min=hp_l['softmax_temperature_min'],
        )
        self.pi_l = opt.get_policy()
        return opt

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, n_rollouts: int = None, horizon: int = None, seed: int = 0) -> float:
        """
        Evaluate the joint policy (π_g, π_l) by simulation.

        At each step:
          1. Subsample k agents → count vector
          2. π_g picks a_g
          3. Each agent samples a_l ~ π_l
          4. Collect reward r_g(s_g,a_g) + (1/n) Σ r_l(s_i,s_g,a_i)
          5. Transition states

        Returns mean discounted return over rollouts.
        """
        hp_e = HP['evaluation']
        if n_rollouts is None:
            n_rollouts = hp_e['n_rollouts']
        if horizon is None:
            horizon = hp_e['horizon']

        rng = np.random.default_rng(seed)
        n = self.n_agents

        # Precompute local policy as array for fast sampling
        pi_l_cum = np.zeros((self.n_sg, self.n_sl, self.n_al))
        for sg in range(self.n_sg):
            for sl in range(self.n_sl):
                pi_l_cum[sg, sl] = self.pi_l[sg][sl]
        pi_l_cum = np.cumsum(pi_l_cum, axis=-1)

        # Precompute P_l and r_l arrays
        P_l_arr = np.zeros((self.n_sl, self.n_sg, self.n_al, self.n_sl))
        r_l_arr = np.zeros((self.n_sl, self.n_sg, self.n_al))
        for sl in range(self.n_sl):
            for sg in range(self.n_sg):
                for al in range(self.n_al):
                    P_l_arr[sl, sg, al] = self.P_l(sl, sg, al)
                    r_l_arr[sl, sg, al] = self.r_l(sl, sg, al)

        P_g_arr = np.zeros((self.n_sg, self.n_ga, self.n_sg))
        r_g_arr = np.zeros((self.n_sg, self.n_ga))
        for sg in range(self.n_sg):
            for ag in range(self.n_ga):
                P_g_arr[sg, ag] = self.P_g(sg, ag)
                r_g_arr[sg, ag] = self.r_g(sg, ag)

        returns = []
        for _ in range(n_rollouts):
            s_g = rng.integers(self.n_sg)
            # Concentrated initial distribution (Dirichlet α=0.3)
            # creates episodes where 1-2 states dominate → mode ID matters
            probs_init = rng.dirichlet(np.full(self.n_sl, 0.3))
            s_agents = rng.choice(self.n_sl, size=n, p=probs_init)
            total = 0.0
            discount = 1.0

            for t in range(horizon):
                # 1. Subsample k agents → count vector
                idx = rng.choice(n, size=self.k, replace=False)
                sampled = s_agents[idx]
                counts = np.bincount(sampled, minlength=self.n_sl)
                counts_key = tuple(counts)

                # 2. Global action
                a_g = self.pi_g.get(s_g, {}).get(counts_key, 0)

                # 3. Local actions (vectorized sampling)
                cum = pi_l_cum[s_g, s_agents]  # (n, n_al)
                u = rng.random((n, 1))
                a_agents = (u >= cum).sum(axis=1).astype(int)
                a_agents = np.clip(a_agents, 0, self.n_al - 1)

                # 4. Reward (vectorized)
                reward = r_g_arr[s_g, a_g] + r_l_arr[s_agents, s_g, a_agents].mean()
                total += discount * reward
                discount *= self.gamma

                # 5. Transitions (vectorized)
                s_g = int(rng.choice(self.n_sg, p=P_g_arr[s_g, a_g]))
                # Local: sample from cumulative distribution
                probs = P_l_arr[s_agents, s_g, a_agents]  # (n, n_sl)
                cum_p = np.cumsum(probs, axis=1)
                u = rng.random((n, 1))
                s_agents = (u >= cum_p).sum(axis=1).astype(int)
                s_agents = np.clip(s_agents, 0, self.n_sl - 1)

            returns.append(total)

        return float(np.mean(returns))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self) -> Dict:
        """
        Alternating best-response training.

        Returns dict with training history.
        """
        hp_a = HP['alternating']
        n_iters = hp_a['n_outer_iterations']
        rtol = hp_a['convergence_rtol']

        history = {'values': [], 'times': []}
        best_value = -np.inf
        best_pi_l = None
        best_pi_g = None

        for it in range(n_iters):
            t0 = time.time()

            # G-LEARN
            self.g_learn()

            # L-LEARN
            self.l_learn()

            # Cheap evaluation for convergence check (20 rollouts)
            val = self.evaluate(n_rollouts=20, horizon=30, seed=it)
            elapsed = time.time() - t0
            history['values'].append(val)
            history['times'].append(elapsed)

            if self.verbose:
                print(f"  iter {it+1}/{n_iters}  value={val:.4f}  time={elapsed:.3f}s")

            # Convergence / revert check
            if val > best_value:
                best_value = val
                best_pi_l = {sg: {sl: p.copy() for sl, p in d.items()}
                             for sg, d in self.pi_l.items()}
                best_pi_g = {sg: dict(d) for sg, d in self.pi_g.items()}
            elif val < best_value - abs(best_value) * rtol:
                # Revert to best
                if best_pi_l is not None:
                    self.pi_l = best_pi_l
                    self.pi_g = best_pi_g
                if self.verbose:
                    print(f"  reverted to best value={best_value:.4f}")

            # Check convergence
            if len(history['values']) >= 2:
                prev = history['values'][-2]
                if abs(val - prev) < rtol * max(abs(prev), 1e-6):
                    if self.verbose:
                        print(f"  converged at iter {it+1}")
                    break

        return history

"""
L-LEARN: Local agent optimizer via value iteration.

Under a fixed global policy π_g, the representative local agent's MDP is:
  State:  (s_g, s_l)  plus k-1 simulated companion agents
  Action: a_l ∈ {0, ..., A_l - 1}
  Reward: r_l(s_l, s_g, a_l)
  Trans:  s_l' ~ P_l(·|s_l, s_g, a_l)
          s_g' determined by π_g applied to (s_g, count_vector) where
              count_vector = representative's state + k-1 companions
          companions transition independently under π_l

Since the model is known, we use value iteration on the (s_g, s_l) MDP,
marginalizing over the global transition induced by π_g + mean-field.
The result is a stochastic policy π_l(·|s_l, s_g) (softmax over Q-values).
"""

import numpy as np
from typing import Callable, Dict, Tuple


class LocalAgentOptimizer:
    """
    Model-based value iteration for the local agent.

    The effective global transition is:
        P_g_eff(s_g'|s_g) = E_{counts ~ Multinomial(k, μ)} [ P_g(s_g'|s_g, π_g(s_g, counts)) ]
    estimated via Monte Carlo sampling of count vectors from μ.

    Parameters
    ----------
    n_sg, n_sl, n_al : int
        Dimensions of global states, local states, local actions.
    k : int
        Subsample size (used to sample count vectors for π_g evaluation).
    gamma : float
        Discount factor.
    P_l : callable (s_l, s_g, a_l) -> np.ndarray of shape (n_sl,)
        Local transition probabilities.
    P_g : callable (s_g, a_g) -> np.ndarray of shape (n_sg,)
        Global transition probabilities.
    r_l : callable (s_l, s_g, a_l) -> float
        Local reward function.
    pi_g : dict  {s_g: {counts_tuple: a_g}}
        Current global policy (fixed during L-LEARN).
    mu : np.ndarray of shape (n_sl,)
        Mean-field distribution over local states.
    max_iter, tol : int, float
        VI convergence parameters.
    tau_scale, tau_min : float
        Softmax temperature: τ = max(tau_min, q_range / tau_scale).
    """

    def __init__(
        self,
        n_sg: int,
        n_sl: int,
        n_al: int,
        k: int,
        gamma: float,
        P_l: Callable,
        P_g: Callable,
        r_l: Callable,
        pi_g: Dict,
        mu: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-6,
        tau_scale: float = 3.0,
        tau_min: float = 0.3,
    ):
        self.n_sg = n_sg
        self.n_sl = n_sl
        self.n_al = n_al

        # --- precompute local transition & reward tables ---
        # P_l_arr[s_g, s_l, a_l, s_l'] and R_arr[s_g, s_l, a_l]
        P_l_arr = np.zeros((n_sg, n_sl, n_al, n_sl))
        R_arr = np.zeros((n_sg, n_sl, n_al))
        for sg in range(n_sg):
            for sl in range(n_sl):
                for al in range(n_al):
                    P_l_arr[sg, sl, al] = P_l(sl, sg, al)
                    R_arr[sg, sl, al] = r_l(sl, sg, al)

        # --- estimate effective global transition via MC ---
        # P_g_eff[s_g, s_g'] = E_{d~Mult(k,μ)}[ P_g(s_g'|s_g, π_g(s_g, d)) ]
        rng = np.random.default_rng(999)
        n_mc = 200
        P_g_eff = np.zeros((n_sg, n_sg))
        for sg in range(n_sg):
            for _ in range(n_mc):
                counts = tuple(rng.multinomial(k, mu))
                a_g = pi_g.get(sg, {}).get(counts, 0)
                P_g_eff[sg] += P_g(sg, a_g)
            P_g_eff[sg] /= n_mc

        # --- vectorized value iteration ---
        V = np.zeros((n_sg, n_sl))
        for _ in range(max_iter):
            # weighted_V[s_l] = Σ_{s_g'} P_g_eff[s_g, s_g'] * V[s_g', s_l]
            # Q[s_g, s_l, a_l] = R[s_g,s_l,a_l] + γ * P_l[s_g,s_l,a_l,:] @ weighted_V
            Q_all = np.zeros((n_sg, n_sl, n_al))
            for sg in range(n_sg):
                weighted_V = P_g_eff[sg] @ V  # (n_sl,)
                Q_all[sg] = R_arr[sg] + gamma * (P_l_arr[sg] @ weighted_V)
            V_new = Q_all.max(axis=2)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        # --- recompute Q at convergence ---
        self.Q = np.zeros((n_sg, n_sl, n_al))
        for sg in range(n_sg):
            weighted_V = P_g_eff[sg] @ V
            self.Q[sg] = R_arr[sg] + gamma * (P_l_arr[sg] @ weighted_V)

        # --- extract softmax policy ---
        self.policy = {}  # policy[s_g][s_l] -> np.ndarray (distribution over a_l)
        for sg in range(n_sg):
            self.policy[sg] = {}
            for sl in range(n_sl):
                q = self.Q[sg, sl]
                q_range = q.max() - q.min()
                tau = max(tau_min, q_range / tau_scale)
                shifted = (q - q.max()) / tau
                exp_q = np.exp(shifted)
                self.policy[sg][sl] = exp_q / exp_q.sum()

    def get_policy(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Return π_l[s_g][s_l] -> probability distribution over actions."""
        return self.policy

    def get_action(self, s_g: int, s_l: int, rng: np.random.Generator = None) -> int:
        """Sample an action from the stochastic policy."""
        p = self.policy[s_g][s_l]
        if rng is None:
            return int(np.random.choice(len(p), p=p))
        return int(rng.choice(len(p), p=p))

"""
G-LEARN: Global agent optimizer via sample-based value iteration.

Under a fixed local policy π_l, the global agent's MDP is:
  State:  (s_g, count_vector)  where count_vector ∈ ℤ_+^L, Σ = k
  Action: a_g ∈ {0, ..., A_g - 1}
  Reward: r_g(s_g, a_g) + (1/k) Σ_j counts[j] · r̃_l(j, s_g)
  Trans:  s_g' ~ P_g(·|s_g, a_g),  counts' from independent local transitions

The count-vector transition does NOT depend on a_g, so we pre-sample
next count vectors once per (s_g, dist) and reuse across actions.

Speed: vectorized multinomial sampling + numpy lookup table for
count-vector → index conversion.  Handles k=20 in ~1–2 seconds.
"""

import numpy as np
from itertools import combinations_with_replacement
from typing import Callable, Dict, Tuple, List


def enumerate_count_vectors(k: int, n_states: int) -> List[Tuple[int, ...]]:
    """All count vectors with k agents across n_states. Returns sorted list of tuples."""
    seen = set()
    result = []
    for combo in combinations_with_replacement(range(n_states), k):
        counts = [0] * n_states
        for s in combo:
            counts[s] += 1
        t = tuple(counts)
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


class GlobalAgentOptimizer:
    """
    Sample-based value iteration for the global agent's (s_g, count_vector) MDP.

    Parameters
    ----------
    n_sg, n_sl, n_ga : int
        Number of global states, local states, global actions.
    k : int
        Subsample size.
    gamma : float
        Discount factor.
    P_g : callable (s_g, a_g) -> np.ndarray of shape (n_sg,)
        Global transition probabilities.
    P_l_pi : callable (s_l, s_g) -> np.ndarray of shape (n_sl,)
        Local transition under fixed π_l  (marginalized over a_l).
    r_g : callable (s_g, a_g) -> float
        Global reward.
    r_l_pi : callable (s_l, s_g) -> float
        Local reward under fixed π_l  (marginalized over a_l).
    n_mc : int
        Number of Monte-Carlo samples for estimating count-vector transitions.
    max_iter : int
        Maximum VI iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        n_sg: int,
        n_sl: int,
        n_ga: int,
        k: int,
        gamma: float,
        P_g: Callable,
        P_l_pi: Callable,
        r_g: Callable,
        r_l_pi: Callable,
        n_mc: int = 20,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        self.n_sg = n_sg
        self.n_sl = n_sl
        self.n_ga = n_ga
        self.k = k
        self.gamma = gamma
        self.n_mc = n_mc

        # --- precompute arrays ---
        self.P_g_arr = np.zeros((n_sg, n_ga, n_sg))
        for sg in range(n_sg):
            for ag in range(n_ga):
                self.P_g_arr[sg, ag] = P_g(sg, ag)

        self.P_l_pi_arr = np.zeros((n_sl, n_sg, n_sl))
        for sl in range(n_sl):
            for sg in range(n_sg):
                self.P_l_pi_arr[sl, sg] = P_l_pi(sl, sg)

        self.R_g = np.zeros((n_sg, n_ga))
        for sg in range(n_sg):
            for ag in range(n_ga):
                self.R_g[sg, ag] = r_g(sg, ag)

        self.R_l_pi = np.zeros((n_sl, n_sg))
        for sl in range(n_sl):
            for sg in range(n_sg):
                self.R_l_pi[sl, sg] = r_l_pi(sl, sg)

        # --- enumerate count vectors + fast lookup table ---
        self.dist_list = enumerate_count_vectors(k, n_sl)
        self.n_dists = len(self.dist_list)
        self.dist_to_idx = {d: i for i, d in enumerate(self.dist_list)}
        self.dist_arr = np.array(self.dist_list, dtype=np.float64)  # (n_dists, n_sl)
        self._dist_arr_int = np.array(self.dist_list, dtype=np.int32)

        # numpy lookup table: shape (k+1)^n_sl, maps count tuple → flat index
        self._lookup = np.full([k + 1] * n_sl, 0, dtype=np.int32)
        for i, d in enumerate(self.dist_list):
            self._lookup[d] = i

        # --- precompute reward table ---
        self.R_table = np.zeros((n_sg, self.n_dists, n_ga))
        for sg in range(n_sg):
            local_r = self.dist_arr @ self.R_l_pi[:, sg] / k
            for ag in range(n_ga):
                self.R_table[sg, :, ag] = self.R_g[sg, ag] + local_r

        # --- deterministic mean-field transition ---
        self._compute_deterministic_transitions()

        # --- value iteration ---
        self.V = np.zeros((n_sg, self.n_dists))
        self.Q = np.zeros((n_sg, self.n_dists, n_ga))
        self._run_vi(max_iter, tol)

    # ------------------------------------------------------------------
    def _round_to_counts(self, expected: np.ndarray) -> np.ndarray:
        """
        Round real-valued vectors to integer count vectors summing to k.
        Uses largest-remainder method (vectorized).

        Parameters
        ----------
        expected : ndarray of shape (n, n_sl), each row sums to k (float).

        Returns
        -------
        result : ndarray of shape (n, n_sl), integer, each row sums to k.
        """
        k = self.k
        floored = np.floor(expected).astype(np.int32)
        residuals = expected - floored
        deficit = k - floored.sum(axis=1)  # (n,) — how many +1s needed

        # Rank columns by descending residual
        rank = np.argsort(np.argsort(-residuals, axis=1), axis=1)
        # Add 1 to the top-`deficit` columns per row
        add_mask = rank < deficit[:, None]
        return floored + add_mask.astype(np.int32)

    def _compute_deterministic_transitions(self):
        """
        Deterministic mean-field transition: E[next_counts] = dist @ P_l_pi.
        Rounded to nearest valid count vector via largest-remainder method.
        Result: self.det_next_idx of shape (n_sg, n_dists).
        """
        n_sg, n_sl = self.n_sg, self.n_sl
        self.det_next_idx = np.zeros((n_sg, self.n_dists), dtype=np.int32)

        for sg in range(n_sg):
            # P_l_pi[:, sg] has shape (n_sl, n_sl): row j = transition from state j
            P = self.P_l_pi_arr[:, sg]  # (n_sl, n_sl)
            expected = self.dist_arr @ P  # (n_dists, n_sl), rows sum to k
            rounded = self._round_to_counts(expected)
            rounded = np.clip(rounded, 0, self.k)
            idx_tuple = tuple(rounded[:, j] for j in range(n_sl))
            self.det_next_idx[sg] = self._lookup[idx_tuple]

    # ------------------------------------------------------------------
    def _run_vi(self, max_iter: int, tol: float):
        """Bellman backup using deterministic mean-field transitions."""
        for iteration in range(max_iter):
            # Vectorized gather: V[:, det_next_idx] → (n_sg, n_sg, n_dists)
            # Then transpose to (n_sg, n_dists, n_sg) = E_next[sg, d, sg']
            E_next = self.V[:, self.det_next_idx].transpose(1, 2, 0)

            # Q = R + γ * Σ_{sg'} P_g(sg'|sg,ag) * E_next(sg,d,sg')
            future = np.einsum('sdp,sap->sda', E_next, self.P_g_arr)
            Q_new = self.R_table + self.gamma * future
            V_new = Q_new.max(axis=2)

            delta = np.max(np.abs(V_new - self.V))
            self.V = V_new
            self.Q = Q_new
            if delta < tol:
                break

    # ------------------------------------------------------------------
    def get_optimal_action(self, s_g: int, counts: tuple) -> int:
        """Return greedy action for state (s_g, counts)."""
        idx = self.dist_to_idx.get(counts, None)
        if idx is None:
            return 0
        return int(np.argmax(self.Q[s_g, idx]))

    def get_policy_dict(self) -> Dict[int, Dict[tuple, int]]:
        """Return full policy as nested dict: pi_g[s_g][counts] -> a_g."""
        policy = {}
        for sg in range(self.n_sg):
            policy[sg] = {}
            for di, counts in enumerate(self.dist_list):
                policy[sg][counts] = int(np.argmax(self.Q[sg, di]))
        return policy

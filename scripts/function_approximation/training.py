"""
Alternating MARL with function approximation (REINFORCE).

Same structure as the tabular version:
    repeat:
        G-LEARN  — fix π_l, collect episodes, update π_g via REINFORCE
        L-LEARN  — fix π_g, collect episodes, update π_l via REINFORCE

Data collection simulates the full n-agent system; rewards are recorded
from the perspective of the relevant agent (global or one representative
local agent).
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

from continuous_env import ContinuousWarehouseEnv
from agents import GlobalAgentFA, LocalAgentFA, LocalAgentPPO, LocalAgentTRPO


class AlternatingMARLFA:
    """
    Alternating best-response with REINFORCE policy-gradient agents.

    Parameters
    ----------
    env : ContinuousWarehouseEnv
    k : int   — sub-sample budget
    gamma : float
    """

    def __init__(self, env: ContinuousWarehouseEnv, k: int, gamma: float = 0.95,
                 hidden: int = 64, lr: float = 3e-3,
                 local_algo: str = 'trpo'):
        self.env = env
        self.k = k
        self.gamma = gamma
        self.local_algo = local_algo

        self.g_agent = GlobalAgentFA(
            n_bins=env.n_bins, n_ga=env.n_ga,
            gamma=gamma, lr=lr, hidden=hidden,
        )
        local_cls = {'a2c': LocalAgentFA,
                     'ppo': LocalAgentPPO,
                     'trpo': LocalAgentTRPO}[local_algo]
        self.l_agent = local_cls(
            n_al=env.n_al, gamma=gamma, lr=lr, hidden=hidden,
        )

    # ------------------------------------------------------------------
    # Global agent training  (supervised cross-entropy)
    # ------------------------------------------------------------------
    def _train_global(self, n_samples: int, rng: np.random.Generator,
                      n_epochs: int = 15):
        """Train global agent via supervised learning on
        (histogram_k, true_mode) pairs — analogous to the model-based
        value-iteration step in the tabular version."""
        obs_list, label_list = [], []
        for _ in range(n_samples):
            s_g, s_agents = self.env.reset(rng)
            true_mode = self.env._initial_mode
            hist = self.env.subsample_histogram(s_agents, self.k, rng)
            obs = self.g_agent._make_obs(s_g, hist)
            obs_list.append(obs)
            label_list.append(true_mode)

        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        label_t = torch.from_numpy(np.array(label_list, dtype=np.int64))

        total_loss = 0.0
        for _ in range(n_epochs):
            logits = self.g_agent.policy.logits(obs_t)
            loss = F.cross_entropy(logits, label_t)
            self.g_agent.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.g_agent.policy.parameters(), 1.0)
            self.g_agent.optimiser.step()
            total_loss += loss.item()

        return total_loss / n_epochs

    def _collect_local(self, n_episodes: int, horizon: int,
                       rng: np.random.Generator):
        """Collect episodes from a representative local agent's perspective."""
        for _ in range(n_episodes):
            s_g, s_agents = self.env.reset(rng)
            rep = rng.integers(self.env.n_agents)
            self.l_agent.buffer.start_episode()

            for _t in range(horizon):
                s_l = float(s_agents[rep])
                obs_l = np.array([s_g, s_l], dtype=np.float32)

                # Global action (deterministic — π_g is fixed)
                hist = self.env.subsample_histogram(s_agents, self.k, rng)
                a_g = self.g_agent.get_action(s_g, hist, deterministic=True)

                # Representative agent samples action from π_l
                a_l_rep = self.l_agent.get_action(s_g, s_l)

                # All other agents follow current π_l
                a_l_all = self.l_agent.get_actions_vec(s_g, s_agents)
                a_l_all[rep] = a_l_rep

                # Reward for representative (canonical r_l(s_l, s_g, a_l))
                r_l = float(self.env.reward_local_vec(
                    np.array([s_l]), s_g, np.array([a_l_rep]))[0])

                self.l_agent.buffer.add(obs_l, a_l_rep, r_l)

                # Transitions
                s_g = self.env.step_global(s_g, a_g, rng)
                s_agents = self.env.step_local_vec(s_agents, s_g, a_l_all, rng)

            self.l_agent.buffer.end_episode()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, n_outer: int = 10, n_samples_g: int = 3000,
              n_collect_l: int = 120, horizon: int = 40,
              n_epochs_g: int = 20, n_epochs_l: int = 5,
              verbose: bool = False):
        """Alternating best-response training."""
        rng = np.random.default_rng(42)
        history = {'values': [], 'times': []}

        for it in range(n_outer):
            t0 = time.time()

            # G-LEARN  (supervised cross-entropy)
            loss_g = self._train_global(n_samples_g, rng,
                                        n_epochs=n_epochs_g)

            # L-LEARN  (REINFORCE)
            self._collect_local(n_collect_l, horizon, rng)
            loss_l = self.l_agent.update(n_epochs=n_epochs_l)

            # Quick evaluation
            val = self.evaluate(n_rollouts=20, horizon=40, seed=it)
            elapsed = time.time() - t0
            history['values'].append(val)
            history['times'].append(elapsed)

            if verbose:
                print(f"  iter {it+1}/{n_outer}  value={val:.4f}  "
                      f"loss_g={loss_g:.4f}  loss_l={loss_l:.4f}  "
                      f"time={elapsed:.1f}s")

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, n_rollouts: int = 50, horizon: int = 50,
                 seed: int = 0) -> float:
        """Mean discounted return over rollouts (deterministic policies)."""
        rng = np.random.default_rng(seed)
        returns = []

        for _ in range(n_rollouts):
            s_g, s_agents = self.env.reset(rng)
            total = 0.0
            discount = 1.0

            for _t in range(horizon):
                hist = self.env.subsample_histogram(s_agents, self.k, rng)
                a_g = self.g_agent.get_action(s_g, hist, deterministic=True)
                a_l = self.l_agent.get_actions_vec(s_g, s_agents,
                                                    deterministic=True)

                r = (self.env.reward_global(s_g, a_g)
                     + self.env.reward_local_vec(s_agents, s_g, a_l).mean())
                total += discount * r
                discount *= self.gamma

                s_g = self.env.step_global(s_g, a_g, rng)
                s_agents = self.env.step_local_vec(s_agents, s_g, a_l, rng)

            returns.append(total)

        return float(np.mean(returns))

    def evaluate_detailed(self, horizon: int = 75, seed: int = 0) -> dict:
        """Single-episode per-step recording (for heatmap visualisation)."""
        rng = np.random.default_rng(seed)
        env = self.env
        s_g, s_agents = env.reset(rng)

        sg_trace = np.zeros(horizon)
        dispatcher_bin = np.zeros(horizon, dtype=int)
        true_mode_bin = np.zeros(horizon, dtype=int)
        histograms = np.zeros((horizon, env.n_bins))
        rewards = np.zeros(horizon)
        mode_correct = np.zeros(horizon, dtype=int)

        for t in range(horizon):
            # Record
            full_hist = env.full_histogram(s_agents)
            histograms[t] = full_hist
            true_mode_bin[t] = env._initial_mode
            sg_trace[t] = s_g

            # Act
            hist_k = env.subsample_histogram(s_agents, self.k, rng)
            a_g = self.g_agent.get_action(s_g, hist_k, deterministic=True)
            dispatcher_bin[t] = a_g
            mode_correct[t] = int(a_g == true_mode_bin[t])

            a_l = self.l_agent.get_actions_vec(s_g, s_agents,
                                                deterministic=True)

            rewards[t] = (env.reward_global(s_g, a_g)
                          + env.reward_local_vec(s_agents, s_g, a_l).mean())

            # Transition
            s_g = env.step_global(s_g, a_g, rng)
            s_agents = env.step_local_vec(s_agents, s_g, a_l, rng)

        return {
            'histograms': histograms,
            'dispatcher_bin': dispatcher_bin,
            'true_mode_bin': true_mode_bin,
            'sg_trace': sg_trace,
            'rewards': rewards,
            'mode_correct': mode_correct,
        }

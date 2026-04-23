"""
Policy-gradient agents for the continuous-state MARL setting.

GlobalAgentFA  — state = (s_g, histogram[0..B-1])  →  action probs for n_ga actions
LocalAgentFA   — state = (s_g, s_l)                →  action probs for n_al actions

Both use REINFORCE with return-normalisation and entropy bonus.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building blocks
# ======================================================================

class PolicyNetwork(nn.Module):
    """MLP that outputs action log-probabilities."""
    def __init__(self, input_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def logits(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """MLP that outputs a scalar state value V(s)."""
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EpisodeBuffer:
    """Stores complete episodes for REINFORCE."""
    def __init__(self):
        self.episodes = []
        self._cur = None

    def start_episode(self):
        self._cur = {'obs': [], 'actions': [], 'rewards': []}

    def add(self, obs, action, reward):
        self._cur['obs'].append(obs)
        self._cur['actions'].append(action)
        self._cur['rewards'].append(reward)

    def end_episode(self):
        self.episodes.append(self._cur)
        self._cur = None

    def compute_returns(self, gamma):
        all_obs, all_act, all_ret = [], [], []
        for ep in self.episodes:
            T = len(ep['rewards'])
            G = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running = ep['rewards'][t] + gamma * running
                G[t] = running
            all_obs.extend(ep['obs'])
            all_act.extend(ep['actions'])
            all_ret.extend(G)
        return (np.array(all_obs, dtype=np.float32),
                np.array(all_act, dtype=np.int64),
                np.array(all_ret, dtype=np.float32))

    def clear(self):
        self.episodes = []

    def __len__(self):
        return sum(len(ep['obs']) for ep in self.episodes)


# ======================================================================
# Global agent  (input dim = 1 + n_bins)
# ======================================================================

class GlobalAgentFA:
    """
    REINFORCE global agent.

    Observation: (s_g, hist_0, …, hist_{B-1})  where hist sums to 1.
    """

    def __init__(self, n_bins, n_ga, gamma=0.95, lr=3e-3, hidden=64,
                 entropy_coeff=0.01):
        self.n_ga = n_ga
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.input_dim = n_bins

        self.policy = PolicyNetwork(self.input_dim, n_ga, hidden)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = EpisodeBuffer()

    # ---------- observation ----------
    def _make_obs(self, s_g, hist):
        return np.array(hist, dtype=np.float32)

    # ---------- action selection ----------
    def get_action(self, s_g, hist, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(self._make_obs(s_g, hist)).unsqueeze(0)
            probs = self.policy(obs)[0].numpy()
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_ga, p=probs))

    # ---------- training ----------
    def update(self, n_epochs=3):
        if len(self.buffer) == 0:
            return 0.0
        obs, act, ret = self.buffer.compute_returns(self.gamma)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        ret_t = torch.from_numpy(ret)
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(n_epochs):
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(act_t)

            loss = -(log_probs * ret_t).mean()
            loss -= self.entropy_coeff * dist.entropy().mean()

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / n_epochs


# ======================================================================
# Local agent  (input dim = 2)
# ======================================================================

class LocalAgentFA:
    """
    Advantage Actor-Critic (A2C) local agent.

    Observation: (s_g, s_l).

    Uses a value network V(s_g, s_l) as a baseline to reduce the variance
    of the Monte Carlo policy gradient.  Advantage estimator:
        A_t = G_t - V(o_t)
    where G_t is the Monte Carlo discounted return.  The policy is updated
    with A_t (detached) while the critic is fit to G_t via MSE.
    """

    def __init__(self, n_al, gamma=0.95, lr=3e-3, hidden=64,
                 entropy_coeff=0.01, value_coeff=0.5):
        self.n_al = n_al
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        self.policy = PolicyNetwork(2, n_al, hidden)
        self.value = ValueNetwork(2, hidden)
        self.optimiser = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )
        self.buffer = EpisodeBuffer()

    # ---------- single-agent action ----------
    def get_action(self, s_g, s_l, deterministic=False):
        with torch.no_grad():
            obs = torch.FloatTensor([[s_g, s_l]])
            probs = self.policy(obs)[0].numpy()
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_al, p=probs))

    # ---------- vectorised stochastic policy ----------
    def get_actions_vec(self, s_g, s_l_arr, deterministic=False):
        """Sample actions for all agents.  s_l_arr: (n,)."""
        n = len(s_l_arr)
        with torch.no_grad():
            obs = torch.FloatTensor(
                np.column_stack([np.full(n, s_g, dtype=np.float32),
                                 s_l_arr.astype(np.float32)])
            )
            probs = self.policy(obs).numpy()  # (n, n_al)
        if deterministic:
            return np.argmax(probs, axis=1)
        cum = np.cumsum(probs, axis=1)
        u = np.random.random((n, 1))
        actions = (u >= cum).sum(axis=1).astype(int)
        return np.clip(actions, 0, self.n_al - 1)

    # ---------- training ----------
    def update(self, n_epochs=3):
        if len(self.buffer) == 0:
            return 0.0
        obs, act, ret = self.buffer.compute_returns(self.gamma)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        ret_t = torch.from_numpy(ret)

        total_loss = 0.0
        for _ in range(n_epochs):
            # Critic forward
            v_pred = self.value(obs_t)                  # (N,)
            # Advantage (detached so policy grad doesn't flow into critic)
            adv = (ret_t - v_pred.detach())
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Actor forward
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(act_t)

            actor_loss = -(log_probs * adv).mean()
            entropy_loss = -dist.entropy().mean()
            critic_loss = F.mse_loss(v_pred, ret_t)

            loss = (actor_loss
                    + self.entropy_coeff * entropy_loss
                    + self.value_coeff * critic_loss)

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters())
                + list(self.value.parameters()), 1.0)
            self.optimiser.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / n_epochs


# ======================================================================
# Local agent — PPO variant  (clipped surrogate + value baseline)
# ======================================================================

class LocalAgentPPO(LocalAgentFA):
    """
    Proximal Policy Optimisation (clipped objective) local agent.

    Same observation / action interface as LocalAgentFA.  Differs only in
    the update rule: multiple epochs on the same trajectory using the
    clipped surrogate objective to bound the policy step size.
    """

    def __init__(self, n_al, gamma=0.95, lr=3e-3, hidden=64,
                 entropy_coeff=0.01, value_coeff=0.5, clip_eps=0.2):
        super().__init__(n_al, gamma=gamma, lr=lr, hidden=hidden,
                         entropy_coeff=entropy_coeff, value_coeff=value_coeff)
        self.clip_eps = clip_eps

    def update(self, n_epochs=5):
        if len(self.buffer) == 0:
            return 0.0
        obs, act, ret = self.buffer.compute_returns(self.gamma)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        ret_t = torch.from_numpy(ret)

        with torch.no_grad():
            old_probs = self.policy(obs_t)
            old_log_probs = torch.distributions.Categorical(old_probs).log_prob(act_t)

        total_loss = 0.0
        for _ in range(n_epochs):
            v_pred = self.value(obs_t)
            adv = (ret_t - v_pred.detach())
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(act_t)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(v_pred, ret_t)
            entropy_loss = -dist.entropy().mean()

            loss = (actor_loss
                    + self.value_coeff * critic_loss
                    + self.entropy_coeff * entropy_loss)

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters())
                + list(self.value.parameters()), 1.0)
            self.optimiser.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / n_epochs


# ======================================================================
# Local agent — TRPO variant  (natural gradient + KL trust region)
# ======================================================================

def _flat_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def _set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def _flat_grad(loss, params, create_graph=False, retain_graph=None):
    grads = torch.autograd.grad(loss, params,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                allow_unused=True)
    out = []
    for g, p in zip(grads, params):
        out.append(torch.zeros_like(p).reshape(-1)
                   if g is None else g.reshape(-1))
    return torch.cat(out)


def _conjugate_gradient(Ax, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Ax(p)
        alpha = rr / (p @ Ap + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rr = r @ r
        if new_rr < tol:
            break
        p = r + (new_rr / rr) * p
        rr = new_rr
    return x


class LocalAgentTRPO(LocalAgentFA):
    """
    Trust Region Policy Optimisation local agent.

    Computes the natural-gradient policy step under a KL-divergence trust
    region D_KL(pi_old || pi_new) <= max_kl, solved via conjugate gradient
    plus backtracking line search.  The value function is fit separately
    with MSE using the shared Adam optimiser.
    """

    def __init__(self, n_al, gamma=0.95, lr=3e-3, hidden=64,
                 entropy_coeff=0.0, value_coeff=0.5,
                 max_kl=0.01, cg_iters=10, cg_damping=0.1,
                 line_search_steps=10):
        super().__init__(n_al, gamma=gamma, lr=lr, hidden=hidden,
                         entropy_coeff=entropy_coeff, value_coeff=value_coeff)
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps

    def update(self, n_epochs=5):
        if len(self.buffer) == 0:
            return 0.0
        obs, act, ret = self.buffer.compute_returns(self.gamma)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        ret_t = torch.from_numpy(ret)

        with torch.no_grad():
            v_pred = self.value(obs_t)
        adv = ret_t - v_pred
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        with torch.no_grad():
            old_probs = self.policy(obs_t)
            old_log_probs = torch.distributions.Categorical(old_probs).log_prob(act_t)

        policy_params = list(self.policy.parameters())

        probs = self.policy(obs_t)
        log_probs = torch.distributions.Categorical(probs).log_prob(act_t)
        surr = -(torch.exp(log_probs - old_log_probs) * adv).mean()
        g = -_flat_grad(surr, policy_params, retain_graph=True)

        def Fx(v):
            probs_cur = self.policy(obs_t)
            dist_cur = torch.distributions.Categorical(probs_cur)
            dist_old = torch.distributions.Categorical(old_probs.detach())
            kl = torch.distributions.kl_divergence(dist_old, dist_cur).mean()
            kl_grad = _flat_grad(kl, policy_params, create_graph=True)
            gv = (kl_grad * v).sum()
            hv = _flat_grad(gv, policy_params, retain_graph=True)
            return hv + self.cg_damping * v

        step_dir = _conjugate_gradient(Fx, g, iters=self.cg_iters)
        shs = 0.5 * (step_dir * Fx(step_dir)).sum()
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_size * step_dir

        old_params = _flat_params(self.policy).clone()
        surr_old = surr.detach()
        success = False
        for i in range(self.line_search_steps):
            step = (0.5 ** i) * full_step
            _set_flat_params(self.policy, old_params + step)
            with torch.no_grad():
                new_probs = self.policy(obs_t)
                new_log_probs = torch.distributions.Categorical(new_probs).log_prob(act_t)
                new_surr = -(torch.exp(new_log_probs - old_log_probs) * adv).mean()
                new_kl = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(old_probs),
                    torch.distributions.Categorical(new_probs),
                ).mean()
            if new_surr < surr_old and new_kl < self.max_kl * 1.5:
                success = True
                break
        if not success:
            _set_flat_params(self.policy, old_params)

        for _ in range(n_epochs):
            v_pred = self.value(obs_t)
            v_loss = F.mse_loss(v_pred, ret_t)
            self.optimiser.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.optimiser.step()

        self.buffer.clear()
        return float(surr_old.item())

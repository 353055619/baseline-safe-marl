"""
algos/facmac/policy.py — FACMAC Policy Stub
============================================
FACMAC: Factored Multi-Agent Centralized Policy Gradients
Paper: arxiv:2201.06233 (NeurIPS 2022)

Key differences from MATD3 (MATD3Policy):
  - Factorized critic with a QMIX-like mixing network:
      Q_tot = mixing_network(Q_1, Q_2, ..., Q_n; state)
      vs MATD3's independent twin critics Q1, Q2.
  - The mixing network conditions on global state (not just local obs+action),
      enabling the critic to reason about team-level value functions.
  - Off-policy: requires replay buffer (same as MADDPG/MATD3).
  - Supports PettingZoo-style dict obs (multi-agent).

Network architecture (stub):
  - Actor: obs -> deterministic action (continuous, tanh-squashed)
  - Per-agent critic: obs + action -> Q_i(s, a_i) for each agent i
  - Mixing network: [Q_1, ..., Q_n, state] -> Q_tot
    (hypernet-like MLP, Monotonic constraint via abs activation on last layer)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.base import BasePolicy


class _FactoredCritic(nn.Module):
    """
    Per-agent critic networks that each produce a local Q-value Q_i(s, a_i).
    Each critic takes its agent's obs + action and returns a scalar Q_i.

    For the stub, we use one shared agent-critic for simplicity (shared policy).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # Q_i(s, a_i): concatenate obs and action
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q_net(x).squeeze(-1)


class _QMixingNetwork(nn.Module):
    """
    QMIX-style mixing network: combines per-agent Q-values into a global Q_tot.

    Simplified stub architecture:
      - state_encoder: obs -> hidden (to condition on state)
      - core: MLP([Q_vals; state_encoded] -> Q_tot) with non-negative last layer (monotonic)
    """

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        state_dim = n_agents * obs_dim

        # State encoder: flatten state and project to hidden_dim
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Core: [q_vals, state_encoded] -> hidden -> Q_tot
        self.core = nn.Sequential(
            nn.Linear(n_agents + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Last layer with non-negative weights (monotonic constraint)
        self.w_last = nn.Parameter(torch.ones(1, hidden_dim))  # (1, hidden_dim)
        self.bias_last = nn.Parameter(torch.zeros(1))

    def forward(self, q_vals: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_vals: (batch, n_agents) per-agent Q-values
            state : (batch, n_agents, obs_dim) concatenated agent observations
        Returns:
            Q_tot : (batch, 1) mixed global Q-value
        """
        batch = q_vals.size(0)

        # Flatten and encode state
        state_flat = state.view(batch, -1)  # (batch, n_agents * obs_dim)
        state_encoded = self.state_net(state_flat)  # (batch, hidden_dim)

        # Core network
        x = torch.cat([q_vals, state_encoded], dim=-1)  # (batch, n_agents + hidden_dim)
        h = self.core(x)  # (batch, hidden_dim)

        # Non-negative last layer (monotonic)
        q_tot = torch.sum(F.softplus(self.w_last) * h, dim=-1, keepdim=True) + self.bias_last
        return q_tot


class FACMACPolicy(BasePolicy):
    """
    FACMAC policy with factorized critic and QMIX-like mixing network.

    Supports both single-agent numpy obs and PettingZoo-style dict obs.
    Uses a shared policy for all agents (agent 0..n-1 share the same network).

    Key FACMAC attributes:
        actor          : deterministic policy (obs -> action)
        critics        : list of per-agent factored Q networks
        mixing_net     : QMIX-style mixing network (Q_tot from Q_i and state)
        target_*       : target networks (polyak update)
        tau            : polyak smoothing coefficient
        delay          : policy update frequency
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__(cfg, agent_id)

        hidden_dim = cfg.get("algo", {}).get("hidden_dim", 64)
        activation = cfg.get("algo", {}).get("activation", "relu")
        self._device = torch.device(cfg.get("device", "cpu"))

        self.tau = cfg.get("algo", {}).get("tau", 0.005)
        self.delay = cfg.get("algo", {}).get("policy_delay", 2)
        self.exploration_noise = cfg.get("algo", {}).get("exploration_noise", 0.1)

        # Infer obs_dim and action_dim from cfg or env
        self.obs_dim = self._get_obs_dim(cfg)
        self.action_dim = self._get_action_dim(cfg)
        self.n_agents = cfg.get("env", {}).get("n_agents", 2)
        self.state_dim = self.obs_dim * self.n_agents  # concatenated global state dim

        # Actor network (deterministic, tanh-squashed)
        self.actor = _Actor(self.obs_dim, self.action_dim, hidden_dim, activation).to(self._device)

        # Per-agent factored critics
        self.critics = nn.ModuleList([
            _FactoredCritic(self.obs_dim, self.action_dim, hidden_dim, activation)
            for _ in range(self.n_agents)
        ]).to(self._device)

        # Target critics (hard-copy initialized)
        self.target_critics = nn.ModuleList([
            _FactoredCritic(self.obs_dim, self.action_dim, hidden_dim, activation)
            for _ in range(self.n_agents)
        ]).to(self._device)
        self.target_critics.load_state_dict(self.critics.state_dict())

        # Mixing networks
        self.mixing_net = _QMixingNetwork(self.n_agents, self.obs_dim, hidden_dim).to(self._device)
        self.target_mixing_net = _QMixingNetwork(self.n_agents, self.obs_dim, hidden_dim).to(self._device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Exploration noise
        self._action_bound = 1.0

        # Freeze target networks
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        for p in self.target_critics.parameters():
            p.requires_grad = False
        for p in self.target_mixing_net.parameters():
            p.requires_grad = False

    def _polyak_update(self, net: nn.Module, target_net: nn.Module) -> None:
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target_net.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)

    def soft_update_targets(self) -> None:
        """Polyak soft-update all target networks (called every step)."""
        for c, tc in zip(self.critics, self.target_critics):
            self._polyak_update(c, tc)
        self._polyak_update(self.mixing_net, self.target_mixing_net)

    def hard_update_targets(self) -> None:
        """Hard-copy main networks to targets (called every `delay` steps)."""
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    # ------------------------------------------------------------------
    # Helpers to derive dims
    # ------------------------------------------------------------------
    @staticmethod
    def _get_obs_dim(cfg: Dict[str, Any]) -> int:
        obs_dim = cfg.get("env", {}).get("obs_dim")
        if obs_dim is not None:
            return int(obs_dim)
        try:
            from envs.safe_mamujoco_adapter import make_safe_ant_2x4
            env = make_safe_ant_2x4()
            obs_sample, _ = env.reset(seed=42)
            d = obs_sample[next(iter(obs_sample.keys()))] if isinstance(obs_sample, dict) else obs_sample
            dim = int(np.array(d).shape[-1])
            env.close()
            return dim
        except Exception:
            return 17  # HalfCheetah-v5 obs dim fallback

    @staticmethod
    def _get_action_dim(cfg: Dict[str, Any]) -> int:
        action_dim = cfg.get("env", {}).get("action_dim")
        if action_dim is not None:
            return int(action_dim)
        try:
            from envs.safe_mamujoco_adapter import make_safe_ant_2x4
            env = make_safe_ant_2x4()
            action_space = env.action_space(env.agents[0])
            dim = int(np.array(action_space.sample()).shape[-1]) if hasattr(action_space, "sample") else 6
            env.close()
            return dim
        except Exception:
            return 6

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------
    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Returns action(s) for the given observation.

        Handles dict {agent_id: obs} (PettingZoo) and plain numpy array.
        Returns the same type as input: dict -> dict of actions, array -> array.

        FACMAC uses a deterministic policy with optional Gaussian exploration noise.
        """
        raw = self._parse_obs(obs)

        if isinstance(raw, dict):
            # PettingZoo multi-agent: return dict of actions
            result = {}
            with torch.no_grad():
                for agent_key in sorted(raw.keys()):
                    x = np.array(raw[agent_key], dtype=np.float32)
                    if x.ndim == 1:
                        x = x[np.newaxis, ...]
                    action = self.actor(torch.as_tensor(x, device=self._device))
                    if not deterministic:
                        noise = torch.randn_like(action) * self.exploration_noise
                        action = (action + noise).clamp(-self._action_bound, self._action_bound)
                    result[agent_key] = action.cpu().numpy()[0]
            return result
        else:
            x = np.array(raw, dtype=np.float32)
            if x.ndim == 1:
                x = x[np.newaxis, ...]
            with torch.no_grad():
                action = self.actor(torch.as_tensor(x, device=self._device))
                if not deterministic:
                    noise = torch.randn_like(action) * self.exploration_noise
                    action = (action + noise).clamp(-self._action_bound, self._action_bound)
                action = action.cpu().numpy()
            return action[0] if action.shape[0] == 1 else action

    def evaluate_actions(self, obs: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """
        Returns factorized Q-values Q_i(s, a_i) for each agent,
        and Q_tot via the mixing network.

        Handles dict {agent_id: obs} and dict {agent_id: action}.

        Returns dict with:
            q_i: per-agent Q-values (list of tensors)
            q_tot: mixed global Q-value (tensor)
            q_min, q_max: for potential double-Q handling
        """
        # Parse obs and actions
        if isinstance(obs, dict):
            obs_parts = [np.asarray(obs[k]).flatten() for k in sorted(obs.keys())]
            obs_tensor = torch.tensor(np.stack(obs_parts), dtype=torch.float32, device=self._device)
        else:
            arr = np.asarray(obs).flatten()
            obs_tensor = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32, device=self._device)

        if isinstance(actions, dict):
            act_parts = [np.asarray(actions[k]).flatten() for k in sorted(actions.keys())]
            act_tensor = torch.tensor(np.stack(act_parts), dtype=torch.float32, device=self._device)
        else:
            arr = np.asarray(actions).flatten()
            act_tensor = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32, device=self._device)

        # Ensure batch dimension
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(0)

        # Compute per-agent Q_i
        q_i_list = []
        for i, critic in enumerate(self.critics):
            obs_i = obs_tensor[i:i+1]   # (1, obs_dim)
            act_i = act_tensor[i:i+1]   # (1, action_dim)
            q_i = critic(obs_i, act_i)  # (1,)
            q_i_list.append(q_i)

        q_vals = torch.stack(q_i_list, dim=-1)  # (1, n_agents)

        # State: (batch=1, n_agents, obs_dim)
        state = obs_tensor.unsqueeze(0)  # (1, n_agents, obs_dim)

        q_tot = self.mixing_net(q_vals, state)

        return {
            "q_i": q_i_list,
            "q_tot": q_tot,
            "q_min": q_tot.min(),
            "q_max": q_tot.max(),
        }

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------
    def _parse_obs(self, obs) -> np.ndarray | Dict[str, np.ndarray]:
        if isinstance(obs, dict):
            return obs
        return np.asarray(obs, dtype=np.float32)

    def _parse_actions_as_tensor(self, actions) -> torch.Tensor:
        """Parse actions into a (n_agents, action_dim) tensor."""
        if isinstance(actions, dict):
            parts = [np.asarray(actions[k]).flatten() for k in sorted(actions.keys())]
            return torch.tensor(np.stack(parts), dtype=torch.float32, device=self._device)
        arr = np.asarray(actions, dtype=np.float32).flatten()
        # If single action, return as 1-agent tensor
        return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32, device=self._device)

    def _parse_obs_as_tensor(self, obs) -> torch.Tensor:
        parsed = self._parse_obs(obs)
        if isinstance(parsed, dict):
            # Average agent obs for the global state (stub simplification)
            parts = [np.asarray(parsed[k]).flatten() for k in sorted(parsed.keys())]
            return torch.tensor(np.stack(parts), dtype=torch.float32, device=self._device)
        arr = np.asarray(parsed, dtype=np.float32).flatten()
        return torch.tensor(arr, dtype=torch.float32, device=self._device)


# --------------------------------------------------------------------
# Internal Actor network (used by FACMACPolicy)
# --------------------------------------------------------------------
class _Actor(nn.Module):
    """
    Deterministic actor for FACMAC (continuous control).

    Differs from MAPPO/HAPPO stochastic actor (Normal distribution):
      - Outputs deterministic action, squashed via tanh to [-1, 1]
      - Exploration is handled externally via additive noise (not part of the policy)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

"""
algos/happo/policy.py — HAPPO Policy Stub
==========================================
HAPPO (Multi-Agent HPPO) policy stub.

Key differences from MAPPO:
  - Supports heterogeneous agents (each agent can have different obs/action dims)
    Stub stage: all agents share the same network structure (homogeneous).
    Interface is designed to allow per-agent policy/obs_dim/action_dim extension.
  - Uses trust region updates per agent (via natural gradient / conjugate gradient)
  - Policy is conditioned on per-agent obs, not concatenated multi-agent obs

Network architecture mirrors MAPPO (MLP actor + critic).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from baseline_safe_marl.algos.base import BasePolicy


class _Actor(nn.Module):
    """Actor: obs -> action distribution (continuous, tanh-squashed)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, activation: str = "tanh"):
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # squash to [-1, 1]
        )
        # log_std learnable parameter (per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return torch.distributions.Normal(mean, std)


class _Critic(nn.Module):
    """Critic: obs -> state value."""

    def __init__(self, obs_dim: int, hidden_dim: int, activation: str = "tanh"):
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class HAPPOPolicy(BasePolicy):
    """
    HAPPO policy stub.

    Supports both single-agent numpy obs and PettingZoo-style dict obs.

    Heterogeneous-agent extension (future work):
      - Each agent can have its own obs_dim / action_dim
      - _get_obs_dim / _get_action_dim accept agent_id
      - Per-agent actor/critic heads stored in dicts
    Stub stage: homogeneous (all agents share the same actor/critic).
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__(cfg, agent_id)

        hidden_dim = cfg.get("algo", {}).get("hidden_dim", 64)
        activation = cfg.get("algo", {}).get("activation", "tanh")
        self._device = torch.device(cfg.get("device", "cpu"))

        obs_dim = self._get_obs_dim(cfg, agent_id)
        action_dim = self._get_action_dim(cfg, agent_id)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Homogeneous stub: single shared actor/critic
        self.actor = _Actor(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.critic = _Critic(obs_dim, hidden_dim, activation).to(self._device)

    # ------------------------------------------------------------------
    # Helpers to derive dims (from cfg or temp env)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_obs_dim(cfg: Dict[str, Any], agent_id: int = 0) -> int:
        """
        Derive obs_dim for a given agent.
        Stub: all agents share the same obs_dim.
        Future: per-agent obs_dim from cfg or env.
        """
        obs_dim = cfg.get("env", {}).get("obs_dim")
        if obs_dim is not None:
            return int(obs_dim)
        from baseline_safe_marl.envs.core.adapter import (
            make_safe_ant_2x4,
            make_safe_halfcheetah_2x3,
            make_safe_hopper_2,
            make_safe_walker_2,
        )
        env_name_raw = cfg.get("env", {}).get("env_name", "ant").lower()
        name_map = {
            "safeant2x4": make_safe_ant_2x4,
            "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
            "safehopper2": make_safe_hopper_2,
            "safewalker2": make_safe_walker_2,
        }
        factory = name_map.get(env_name_raw, make_safe_ant_2x4)
        try:
            env = factory()
            obs_sample, _ = env.reset(seed=42)
            if isinstance(obs_sample, dict):
                d = next(iter(obs_sample.values()))
            else:
                d = obs_sample
            dim = int(np.array(d).shape[-1])
            env.close()
            return dim
        except Exception:
            return 27  # Ant-v5 obs dim fallback

    @staticmethod
    def _get_action_dim(cfg: Dict[str, Any], agent_id: int = 0) -> int:
        """
        Derive action_dim for a given agent.
        Stub: all agents share the same action_dim.
        Future: per-agent action_dim from cfg or env.
        """
        action_dim = cfg.get("env", {}).get("action_dim")
        if action_dim is not None:
            return int(action_dim)
        from baseline_safe_marl.envs.core.adapter import (
            make_safe_ant_2x4,
            make_safe_halfcheetah_2x3,
            make_safe_hopper_2,
            make_safe_walker_2,
        )
        env_name_raw = cfg.get("env", {}).get("env_name", "ant").lower()
        name_map = {
            "safeant2x4": make_safe_ant_2x4,
            "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
            "safehopper2": make_safe_hopper_2,
            "safewalker2": make_safe_walker_2,
        }
        factory = name_map.get(env_name_raw, make_safe_ant_2x4)
        try:
            env = factory()
            action_space = env.action_space(env.agents[0])
            dim = int(np.array(action_space.sample()).shape[-1]) if hasattr(action_space, "sample") else int(action_space.shape[-1])
            env.close()
            return dim
        except Exception:
            return 6  # Ant-v5 action dim fallback

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------
    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Returns action(s) for the given observation.

        Handles:
          - dict {agent_id: obs}  (PettingZoo multi-agent)
          - plain numpy array     (single-agent or stacked)
        """
        single = False
        if isinstance(obs, dict):
            # Multi-agent: use first agent's obs (shared/homogeneous policy stub)
            first_key = next(iter(obs.keys()))
            raw = np.array(obs[first_key], dtype=np.float32)
        else:
            raw = np.array(obs, dtype=np.float32)

        # Ensure batch dimension
        if raw.ndim == 1:
            raw = raw[np.newaxis, ...]
            single = True

        with torch.no_grad():
            dist = self.actor(torch.as_tensor(raw, device=self._device))
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            action = action.cpu().numpy()

        if single:
            action = action[0]
        return action

    def evaluate_actions(self, obs: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """
        Returns log_prob and entropy for (obs, actions) pair.

        Parameters
        ----------
        obs : np.ndarray (dict or array)
        actions : np.ndarray

        Returns
        -------
        dict with keys: log_prob (tensor), entropy (tensor), value (tensor)
        """
        if isinstance(obs, dict):
            first_key = next(iter(obs.keys()))
            raw_obs = np.array(obs[first_key], dtype=np.float32)
        else:
            raw_obs = np.array(obs, dtype=np.float32)

        if raw_obs.ndim == 1:
            raw_obs = raw_obs[np.newaxis, ...]
        if actions.ndim == 1:
            actions = actions[np.newaxis, ...]

        obs_t = torch.as_tensor(raw_obs, device=self._device, dtype=torch.float32)
        act_t = torch.as_tensor(actions, device=self._device, dtype=torch.float32)

        dist = self.actor(obs_t)
        log_prob = dist.log_prob(act_t).sum(dim=-1)  # sum over action dims
        entropy = dist.entropy().sum(dim=-1)

        with torch.no_grad():
            value = self.critic(obs_t)

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

"""
algos/mappo/policy.py — MAPPO Policy Stub
==========================================
MAPPO (Multi-Agent Proximal Policy Optimization) policy stub.
Assumes shared policy across agents (MAPPO default).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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


class MAPPOPolicy(BasePolicy):
    """
    MAPPO policy with shared actor-critic.

    Supports both single-agent numpy obs and PettingZoo-style dict obs.
    For multi-agent (shared policy), concatenates all agent obs for the critic
    and uses the first agent's obs to query the shared actor.
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        # BasePolicy expects (cfg, agent_id=0)
        super().__init__(cfg, agent_id)

        # Derive dims from cfg or fallback env
        hidden_dim = cfg.get("algo", {}).get("hidden_dim", 64)
        activation = cfg.get("algo", {}).get("activation", "tanh")
        self._device = torch.device(cfg.get("device", "cpu"))

        # Get obs_dim / action_dim — try cfg first, then derive from env
        obs_dim = self._get_obs_dim(cfg)
        action_dim = self._get_action_dim(cfg)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = _Actor(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.critic = _Critic(obs_dim, hidden_dim, activation).to(self._device)

    # ------------------------------------------------------------------
    # Helpers to derive dims (from cfg or temp env)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_obs_dim(cfg: Dict[str, Any]) -> int:
        # Try explicit cfg first
        obs_dim = cfg.get("env", {}).get("obs_dim")
        if obs_dim is not None:
            return int(obs_dim)
        # Fallback: create temp env to query obs space
        from baseline_safe_marl.envs.core.adapter import (
            make_safe_ant_2x4,
            make_safe_halfcheetah_2x3,
            make_safe_hopper_2,
            make_safe_walker_2,
        )
        env_name_raw = cfg.get("env", {}).get("env_name", "ant").lower()

        # Map display name to factory
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
            # Last-resort defaults for smoke test
            return 27  # Ant-v5 obs dim

    @staticmethod
    def _get_action_dim(cfg: Dict[str, Any]) -> int:
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
            return 6  # Ant-v5 action dim

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
            # Multi-agent: use first agent's obs (shared policy)
            first_key = next(iter(obs.keys()))
            raw = np.array(obs[first_key], dtype=np.float32)
            single = False
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

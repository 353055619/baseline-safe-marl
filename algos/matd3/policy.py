"""
algos/matd3/policy.py — MATD3 Policy Stub
==========================================
MATD3 (Multi-Agent Twin Delayed DDPG) policy stub.

Key differences from MAPPO / HAPPO:
  - Off-policy (DDPG family) vs on-policy (PPO family)
  - Deterministic actor with exploration noise (eps-greedy or Gaussian)
  - Twin critics Q1, Q2 (clipped double Q-learning)
  - Delayed policy updates: policy更新的频率低于Q-function更新
  - Target networks for both actor and twin critics
  - Supports PettingZoo-style dict obs (multi-agent)

Network architecture:
  - Actor: obs -> action (deterministic, tanh-squashed to [-1, 1])
  - Critic: obs + action -> Q(s,a) (two copies: Q1, Q2)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from algos.base import BasePolicy


class _Actor(nn.Module):
    """
    Actor: obs -> deterministic action (continuous, tanh-squashed).

    Unlike MAPPO/HAPPO's stochastic actor (Normal distribution),
    MATD3 uses a deterministic actor + separate exploration strategy.
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
            nn.Tanh(),  # squash to [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _TwinCritic(nn.Module):
    """
    Twin critic: obs + action -> Q(s,a).

    MATD3 uses two separate Q networks (Q1, Q2) and takes the min for
    target computation (clipped double Q, reducing overestimation).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network (twin)
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Q1_value, Q2_value) for (obs, action) pair.
        """
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(x).squeeze(-1)
        q2 = self.q2_net(x).squeeze(-1)
        return q1, q2

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q1 value only (used for policy gradient in actor update)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1_net(x).squeeze(-1)


class MATD3Policy(BasePolicy):
    """
    MATD3 policy with deterministic actor and twin critics.

    Supports both single-agent numpy obs and PettingZoo-style dict obs.
    For multi-agent (shared policy), uses the first agent's obs/action dims.

    Key MATD3 attributes:
        actor       : main deterministic policy
        critic_q1   : twin critic Q1
        critic_q2   : twin critic Q2 (clipped double Q)
        target_actor: target network for actor (delayed polyak update)
        target_q1   : target Q1
        target_q2   : target Q2
        tau         : polyak smoothing coefficient for target updates
        delay       : policy update frequency (steps between actor updates)
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__(cfg, agent_id)

        hidden_dim = cfg.get("algo", {}).get("hidden_dim", 64)
        activation = cfg.get("algo", {}).get("activation", "relu")
        self._device = torch.device(cfg.get("device", "cpu"))

        # Polyak / target network hyperparams
        self.tau = cfg.get("algo", {}).get("tau", 0.005)
        self.delay = cfg.get("algo", {}).get("policy_delay", 2)

        obs_dim = self._get_obs_dim(cfg, agent_id)
        action_dim = self._get_action_dim(cfg, agent_id)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Main networks
        self.actor = _Actor(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.critic_q1 = _TwinCritic(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.critic_q2 = _TwinCritic(obs_dim, action_dim, hidden_dim, activation).to(self._device)

        # Target networks (initialized as copies)
        self.target_actor = _Actor(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.target_q1 = _TwinCritic(obs_dim, action_dim, hidden_dim, activation).to(self._device)
        self.target_q2 = _TwinCritic(obs_dim, action_dim, hidden_dim, activation).to(self._device)

        # Sync target networks with main networks initially
        self._sync_target_networks(hard=True)

        # Exploration noise
        self.exploration_noise = cfg.get("algo", {}).get("exploration_noise", 0.1)

        # Target networks require no gradients
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Target network helpers
    # ------------------------------------------------------------------
    def _sync_target_networks(self, hard: bool = False) -> None:
        """
        Polyak update target networks: θ_target = τ * θ_main + (1-τ) * θ_target.
        If hard=True, hard copy (for initialization).
        """
        if hard:
            # Hard copy
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_q1.load_state_dict(self.critic_q1.state_dict())
            self.target_q2.load_state_dict(self.critic_q2.state_dict())
        else:
            # Polyak soft update
            with torch.no_grad():
                for p_main, p_target in zip(self.actor.parameters(), self.target_actor.parameters()):
                    p_target.data.mul_(1.0 - self.tau)
                    p_target.data.add_(self.tau * p_main.data)
                for p_main, p_target in zip(self.critic_q1.parameters(), self.target_q1.parameters()):
                    p_target.data.mul_(1.0 - self.tau)
                    p_target.data.add_(self.tau * p_main.data)
                for p_main, p_target in zip(self.critic_q2.parameters(), self.target_q2.parameters()):
                    p_target.data.mul_(1.0 - self.tau)
                    p_target.data.add_(self.tau * p_main.data)

    def soft_update_target_networks(self) -> None:
        """Call after each critic update (every step)."""
        self._sync_target_networks(hard=False)

    def hard_update_target_networks(self) -> None:
        """Call periodically (every `delay` steps) after actor update."""
        self._sync_target_networks(hard=True)

    # ------------------------------------------------------------------
    # Helpers to derive dims (from cfg or temp env)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_obs_dim(cfg: Dict[str, Any], agent_id: int = 0) -> int:
        obs_dim = cfg.get("env", {}).get("obs_dim")
        if obs_dim is not None:
            return int(obs_dim)
        from envs.safe_mamujoco_adapter import (
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
            d = obs_sample[next(iter(obs_sample.keys()))] if isinstance(obs_sample, dict) else obs_sample
            dim = int(np.array(d).shape[-1])
            env.close()
            return dim
        except Exception:
            return 27  # Ant-v5 obs dim fallback

    @staticmethod
    def _get_action_dim(cfg: Dict[str, Any], agent_id: int = 0) -> int:
        action_dim = cfg.get("env", {}).get("action_dim")
        if action_dim is not None:
            return int(action_dim)
        from envs.safe_mamujoco_adapter import (
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

        MATD3 uses a deterministic policy with optional exploration noise.
        When deterministic=False, adds Gaussian noise for exploration.
        """
        single = False
        if isinstance(obs, dict):
            first_key = next(iter(obs.keys()))
            raw = np.array(obs[first_key], dtype=np.float32)
        else:
            raw = np.array(obs, dtype=np.float32)

        if raw.ndim == 1:
            raw = raw[np.newaxis, ...]
            single = True

        with torch.no_grad():
            action = self.actor(torch.as_tensor(raw, device=self._device))
            if not deterministic:
                # Add Gaussian exploration noise
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(-1.0, 1.0)
            action = action.cpu().numpy()

        if single:
            action = action[0]
        return action

    def evaluate_actions(self, obs: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """
        Returns Q values from twin critics for (obs, actions) pair.

        Returns
        -------
        dict with keys: q1 (tensor), q2 (tensor), q_min (tensor)
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

        q1_vals, _ = self.critic_q1(obs_t, act_t)
        _, q2_vals = self.critic_q2(obs_t, act_t)
        q_min = torch.min(q1_vals, q2_vals)

        return {
            "q1": q1_vals,
            "q2": q2_vals,
            "q_min": q_min,
        }

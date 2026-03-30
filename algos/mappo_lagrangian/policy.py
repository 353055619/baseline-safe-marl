"""
algos/mappo_lagrangian/policy.py — MAPPO-L Policy Stub
======================================================
Same as MAPPO policy but additionally stores Lagrangian state:
  - self.cost_limit
  - self.lagrangian_multiplier
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from algos.base import BasePolicy


class MAPPOLPolicy(BasePolicy):
    """
    MAPPO-L policy with Lagrangian constraint awareness.

    Stores Lagrangian state from cfg:
      - cost_limit
      - lagrangian_multiplier (lambda)
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__(cfg, agent_id)
        algo_cfg = cfg.get("algo", {})
        self.cost_limit: float = algo_cfg.get("cost_limit", 25.0)
        self.lagrangian_multiplier: float = algo_cfg.get("initial_lagrangian_multiplier", 1.0)

        hidden_dim = algo_cfg.get("hidden_dim", 64)
        activation_str = algo_cfg.get("activation", "tanh")
        self.action_dim: int = algo_cfg.get("action_dim", 1)  # stub: default 1

        activation = nn.Tanh() if activation_str == "tanh" else nn.ReLU()

        # Simple MLP encoder: obs -> hidden -> actor(categorical) / critic
        # Stub: obs_dim inferred from first call or defaulted to 1
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # placeholder; will be resized on first forward
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )

        self.actor_head = nn.Linear(hidden_dim, self.action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.to(self.device)

    def _get_obs_dim(self, obs: np.ndarray) -> int:
        """Infer obs dim from numpy array."""
        if isinstance(obs, dict):
            # PettingZoo-style: flatten first key's value
            first_key = next(iter(obs.keys()))
            val = np.asarray(obs[first_key])
            return int(np.prod(val.shape))
        obs = np.asarray(obs)
        return int(np.prod(obs.shape))

    def _resize_encoder(self, obs_dim: int) -> None:
        """Resize encoder input layer if needed (stub: called once)."""
        if self.encoder[0].in_features == obs_dim:
            return
        self.encoder[0] = nn.Linear(obs_dim, self.encoder[0].out_features)
        self.to(self.device)

    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_dim = self._get_obs_dim(obs)
        self._resize_encoder(obs_dim)

        x = self._obs_to_tensor(obs)
        hidden = self.encoder(x)
        logits = self.actor_head(hidden)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()

        return action.cpu().numpy().astype(np.int64)

    def evaluate_actions(self, obs: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        obs_dim = self._get_obs_dim(obs)
        self._resize_encoder(obs_dim)

        x = self._obs_to_tensor(obs)
        hidden = self.encoder(x)
        logits = self.actor_head(hidden)

        action_tensor = torch.asarray(actions, dtype=torch.long, device=self.device)
        log_prob = torch.distributions.Categorical(logits=logits).log_prob(action_tensor)
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        value = self.critic_head(hidden).squeeze(-1)

        return {
            "log_prob": log_prob.mean().item(),
            "entropy": entropy.mean().item(),
            "value": value.mean().item(),
        }

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        if isinstance(obs, dict):
            # PettingZoo: stack/sum all agent obs
            parts = [np.asarray(obs[k]).flatten() for k in sorted(obs.keys())]
            flat = np.concatenate(parts) if len(parts) > 1 else parts[0]
        else:
            flat = np.asarray(obs).flatten()
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

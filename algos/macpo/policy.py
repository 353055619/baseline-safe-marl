"""
algos/macpo/policy.py — MACPO Policy Stub
==========================================
Multi-Agent Constrained Policy Optimization.

Structure mirrors MAPPOLPolicy but uses PPO-style trust region
(clip objective) for constraint satisfaction instead of Lagrangian.

Key attributes:
  - cost_limit: constraint upper bound per episode
  - encoder: shared MLP
  - actor_head: policy head (Categorical)
  - critic_head: value head
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.base import BasePolicy


class MACPOPolicy(BasePolicy):
    """
    MACPO policy with trust region (PPO-style clip).

    Unlike MAPPO-L which uses Lagrangian multipliers, MACPO enforces
    constraints via trust region and constraint gradient derivation.
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__(cfg, agent_id)
        algo_cfg = cfg.get("algo", {})

        self.cost_limit: float = algo_cfg.get("cost_limit", 25.0)

        hidden_dim = algo_cfg.get("hidden_dim", 64)
        activation_str = algo_cfg.get("activation", "tanh")
        self.action_dim: int = algo_cfg.get("action_dim", 1)

        activation = nn.Tanh() if activation_str == "tanh" else nn.ReLU()

        # MLP encoder: obs -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # placeholder; resized on first forward
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )

        self.actor_head = nn.Linear(hidden_dim, self.action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.clip_eps: float = algo_cfg.get("clip_eps", 0.2)
        self.entropy_coef: float = algo_cfg.get("entropy_coef", 0.01)
        self.value_coef: float = algo_cfg.get("value_coef", 0.5)

        self.to(self.device)

    def _get_obs_dim(self, obs: np.ndarray) -> int:
        """Infer obs dim from numpy array or dict (PettingZoo)."""
        if isinstance(obs, dict):
            first_key = next(iter(obs.keys()))
            val = np.asarray(obs[first_key])
            return int(np.prod(val.shape))
        obs = np.asarray(obs)
        return int(np.prod(obs.shape))

    def _resize_encoder(self, obs_dim: int) -> None:
        """Resize encoder input layer if obs_dim differs from placeholder."""
        if self.encoder[0].in_features == obs_dim:
            return
        self.encoder[0] = nn.Linear(obs_dim, self.encoder[0].out_features)
        self.to(self.device)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        if isinstance(obs, dict):
            parts = [np.asarray(obs[k]).flatten() for k in sorted(obs.keys())]
            flat = np.concatenate(parts) if len(parts) > 1 else parts[0]
        else:
            flat = np.asarray(obs).flatten()
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

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

    def train(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        PPO-style trust region update with clip objective.

        Stub: returns dummy metrics. Real MACPO would compute constraint
        gradient and apply trust region KL projection here.
        """
        logits = self.actor_head(self.encoder(obs))
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean().item()

        # PPO clip objective (stub)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic_head(self.encoder(obs)).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy,
        }

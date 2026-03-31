"""
algos/macpo/trainer.py — MACPO Trainer Stub
============================================
PPO-style trainer for MACPO.

Unlike MAPPO-L which uses Lagrangian multiplier update,
MACPO enforces constraints via trust region + constraint gradient.
update_lagrangian() is a stub that returns current cost (no lambda update).

Optimizer: Adam (PPO-style)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import numpy as np

from algos.base import BasePolicy, BaseTrainer


class MACPOTrainer(BaseTrainer):
    """
    MACPO trainer with PPO-style trust region.

    Constraint satisfaction is handled by trust region clipping,
    not Lagrangian multipliers. The update_lagrangian() stub
    exists for API compatibility and returns the current cost.
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        super().__init__(cfg, policy)
        algo_cfg = cfg.get("algo", {})

        self.cost_limit: float = algo_cfg.get("cost_limit", 25.0)

        lr = algo_cfg.get("lr", 3e-4)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
        )

        self.num_epochs: int = algo_cfg.get("num_epochs", 4)
        self.entropy_coef: float = algo_cfg.get("entropy_coef", 0.01)
        self.value_coef: float = algo_cfg.get("value_coef", 0.5)
        self.clip_eps: float = algo_cfg.get("clip_eps", 0.2)

        self.gamma: float = algo_cfg.get("gamma", 0.99)
        self.gae_lambda: float = algo_cfg.get("gae_lambda", 0.95)

    def train(self, num_steps: int) -> Dict[str, float]:
        """
        Stub training step: PPO-style update with clip objective.

        Real implementation would:
          - Collect rollout data
          - Compute GAE advantages
          - PPO epoch updates with clip
          - (MACPO-specific) apply constraint gradient via trust region

        Returns stub metrics.
        """
        self.total_steps += num_steps

        # Stub metrics
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.1,
            "kl": 0.0,
            "cost": 0.0,
            "cost_limit": self.cost_limit,
        }
        return metrics

    def update_lagrangian(self, current_cost: float = 0.0) -> float:
        """
        Stub for API compatibility.

        MACPO does NOT use Lagrangian multipliers — constraints are handled
        via trust region and constraint gradient derivation. This stub
        simply returns the current cost.

        Returns:
            current_cost (stub)
        """
        return current_cost

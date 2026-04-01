"""
algos/mappo_lagrangian/trainer.py — MAPPO-L Trainer Stub
========================================================
Same as MAPPO trainer but with Lagrangian multiplier update:
  - self.cost_limit
  - self.lagrangian_lr
  - self.lambda_ (Lagrangian multiplier)
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from baseline_safe_marl.algos.base import BasePolicy, BaseTrainer


class MAPPOLTrainer(BaseTrainer):
    """
    MAPPO-L trainer with Lagrangian constraint optimisation.

    Extra attributes beyond BaseTrainer:
      - cost_limit: constraint budget per episode
      - lagrangian_lr: learning rate for lambda update
      - lambda_: current Lagrangian multiplier
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        super().__init__(cfg, policy)
        algo_cfg = cfg.get("algo", {})

        self.cost_limit: float = algo_cfg.get("cost_limit", 25.0)
        self.lagrangian_lr: float = algo_cfg.get("lagrangian_lr", 0.01)
        self.lambda_: float = algo_cfg.get("initial_lagrangian_multiplier", 1.0)

        # Policy optimizer
        lr = algo_cfg.get("lr", 3e-4)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
        )

        # Lagrangian multiplier is a scalar — no gradient-based optimizer needed
        # (updated via simple gradient-ascent rule in update_lagrangian)

        self.num_epochs: int = algo_cfg.get("num_epochs", 4)
        self.entropy_coef: float = algo_cfg.get("entropy_coef", 0.01)
        self.value_coef: float = algo_cfg.get("value_coef", 0.5)
        self.clip_eps: float = algo_cfg.get("clip_eps", 0.2)

    def train(self, num_steps: int) -> Dict[str, float]:
        """
        Stub training step: just increment total_steps and return dummy metrics.
        """
        self.total_steps += num_steps

        # Stub metrics
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.1,
            "kl": 0.0,
            "lambda": self.lambda_,
        }
        return metrics

    def update_lagrangian(self, current_cost: float = 0.0) -> float:
        """
        Update Lagrangian multiplier via simple gradient-ascent rule:

          lambda_ += lagrangian_lr * (current_cost - cost_limit)
          lambda_ = max(0.0, lambda_)

        Returns:
            updated lambda value
        """
        self.lambda_ += self.lagrangian_lr * (current_cost - self.cost_limit)
        self.lambda_ = max(0.0, self.lambda_)
        return self.lambda_

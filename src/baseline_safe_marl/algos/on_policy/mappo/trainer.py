"""
algos/mappo/trainer.py — MAPPO Trainer Stub
============================================
MAPPO trainer stub. Implements BaseTrainer interface.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.optim as optim

from baseline_safe_marl.algos.base import BasePolicy, BaseTrainer


class MAPPOTrainer(BaseTrainer):
    """
    MAPPO trainer stub.

    In a full implementation this would:
      - Collect rollouts via the policy
      - Compute PPO clipped surrogate loss
      - Update actor and critic networks
      - Return training metrics
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        super().__init__(cfg, policy)

        lr = cfg.get("algo", {}).get("lr", 3.0e-4)
        self._optimizer = optim.Adam(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------
    def train(self, num_steps: int) -> Dict[str, float]:
        """
        Stub training step. Increments total_steps and returns dummy metrics.

        In a full MAPPO implementation this would:
          1. Collect `num_steps` of rollout data
          2. Compute PPO loss (policy loss + value loss + entropy bonus)
          3. Perform multiple epochs of minibatch updates
          4. Return metrics dict
        """
        self.total_steps += num_steps

        # Dummy metrics (stub — no real backward pass)
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "lr": self._optimizer.param_groups[0]["lr"],
        }

    def update_lagrangian(self) -> None:
        """
        No-op for MAPPO (non-safe algorithm).
        Present to satisfy BaseTrainer interface.
        """
        pass

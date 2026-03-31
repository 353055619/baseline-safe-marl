"""
algos/happo/trainer.py — HAPPO Trainer Stub
============================================
HAPPO trainer stub. Implements BaseTrainer interface.

HAPPO key ideas (preserved in interface, stub implementation):
  - Per-agent policy updates using natural gradient / trust region
  - HAPPO uses importance sampling and a "permutation-invariant" loss
  - update_lagrangian() is a no-op (HAPPO is not a constrained/safe algorithm)

In a full implementation:
  - Collect rollouts for all agents
  - Compute per-agent advantage using V-trace or GAE
  - For each agent, compute natural gradient via conjugate gradient
  - Apply trust region constrained update
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.optim as optim

from algos.base import BasePolicy, BaseTrainer


class HAPPOTrainer(BaseTrainer):
    """
    HAPPO trainer stub.

    In a full implementation this would:
      - Collect rollouts via the policy (per-agent trajectories)
      - Compute per-agent advantages
      - Perform natural-gradient-based policy updates per agent
      - Apply conjugate gradient to solve trust region subproblem
      - Return per-agent training metrics
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

        In a full HAPPO implementation this would:
          1. Collect `num_steps` of per-agent rollout data
          2. Compute per-agent advantages (GAE or V-trace)
          3. For each agent i:
               - Compute natural gradient using conjugate gradient
               - Apply KL trust-region constrained update
          4. Return per-agent metrics
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
        No-op for HAPPO (non-safe algorithm).
        HAPPO itself does not have constraint / cost layers.
        Present to satisfy BaseTrainer interface.
        """
        pass

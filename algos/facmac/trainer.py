"""
algos/facmac/trainer.py — FACMAC Trainer Stub
==============================================
FACMAC: Factored Multi-Agent Centralized Policy Gradients
Paper: arxiv:2201.06233 (NeurIPS 2022)

Implements BaseTrainer interface.

FACMAC key ideas (preserved in stub comments, not implemented):
  1. Off-policy: replay buffer collects (s, a, r, s', d) tuples
  2. Factorized Q-function: each agent has a local Q_i(s, a_i)
     Q_tot = mixing_network(Q_1..Q_n, state)  (QMIX-style)
  3. Target networks for both critics and mixing network
  4. Clipped double Q: min(Q1_tot, Q2_tot) to reduce overestimation
  5. Delayed policy updates (same as MATD3)
  6. update_lagrangian() is a no-op (FACMAC is not a constrained algorithm;
     FACMAC-Safe would add a CPO/PCPO layer, not in this stub)

In a full implementation:
  1. Sample mini-batch from replay buffer
  2. Compute target actions via target actor (with target policy noise clipping)
  3. Compute target Q_tot via target mixing network:
       y = r + gamma * min(Q1_tot_target, Q2_tot_target)(s', a'_i)
  4. Update per-agent critics + mixing network by minimizing MSE
  5. Every `delay` steps: update actor via deterministic policy gradient
     (maximize Q_tot from the *main* mixing network, not target)
     and hard-update all target networks
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.optim as optim

from algos.base import BasePolicy, BaseTrainer


class FACMACTrainer(BaseTrainer):
    """
    FACMAC trainer stub.

    Maintains the same interface as MATD3Trainer (both are off-policy
    actor-critic methods) but with a factorized critic + mixing network
    instead of independent twin critics.
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        super().__init__(cfg, policy)

        lr = cfg.get("algo", {}).get("lr", 1.0e-3)
        gamma = cfg.get("algo", {}).get("gamma", 0.99)
        self._policy_delay = self.policy.delay
        self._update_counter = 0

        # Optimizers
        self._actor_opt = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self._critic_opt = optim.Adam(
            list(self.policy.mixing_net.parameters())
            + [p for c in self.policy.critics for p in c.parameters()],
            lr=lr,
        )

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------
    def train(self, num_steps: int) -> Dict[str, float]:
        """
        Stub training step.

        In a full FACMAC implementation:
          1. Sample batch from replay buffer:
               (obs_dict, action_dict, reward_dict, next_obs_dict, done_dict)
          2. Critic update (every step):
               - target_actions = clip(target_actor(obs_next_i) + noise, -1, 1)
               - Compute Q_i_target for each agent via target critics
               - Compute Q_tot_target via target mixing network
               - y = r_i + gamma * Q_tot_target(s', a')
               - critic_loss = MSE(Q_i(s, a_i), y_i) for all i + mixing loss
               - step _critic_opt
               - polyak soft-update target critics and mixing_net
          3. Policy update (every `delay` steps):
               - actor_loss = -Q_tot_main(obs, actor(obs))  [deterministic PG]
               - step _actor_opt
               - hard-update all target networks
        """
        self.total_steps += num_steps
        self._update_counter += num_steps

        return {
            "critic_loss": 0.0,
            "mixing_loss": 0.0,
            "actor_loss": 0.0,
            "q_tot": 0.0,
            "policy_delay": self._policy_delay,
            "update_counter": self._update_counter,
        }

    def update_lagrangian(self) -> None:
        """
        No-op for FACMAC (unconstrained algorithm).

        FACMAC itself does not handle safety constraints.
        A constrained extension (e.g., FACMAC-CPO, FACMAC-PCPO) would add
        Lagrangian multiplier updates here.

        Present to satisfy BaseTrainer interface (consistency with MAPPO-L).
        """
        pass

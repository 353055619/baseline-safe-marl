"""
algos/macpo — Multi-Agent Constrained Policy Optimization (Stub)
================================================================
MACPO uses trust region + constraint gradient (not Lagrangian multipliers)
to satisfy safety constraints.

Key differences from MAPPO-L:
  - Trust region (PPO-style clipping) instead of Lagrangian
  - Stores cost_limit for constraint awareness
  - update_lagrangian() is a stub (constraint satisfaction via trust region)
"""

from __future__ import annotations

from .policy import MACPOPolicy  # noqa: F401
from .trainer import MACPOTrainer  # noqa: F401

__all__ = ["MACPOPolicy", "MACPOTrainer"]

"""
algos/matd3/__init__.py — MATD3 Algorithm Stub Package
======================================================
MATD3: Multi-Agent Twin Delayed DDPG.

Exports:
    MATD3Policy
    MATD3Trainer

Key characteristics:
  - Off-policy (DDPG family), not on-policy like MAPPO/HAPPO
  - Deterministic actor with exploration noise
  - Twin critics (Q1, Q2) with clipped double Q-learning
  - Delayed policy updates (policy更新的频率低于critic)
  - Target networks with polyak smoothing

References:
  - MATD3: https://arxiv.org/abs/1910.01465 (original MADDPG extended with twin critics + delay)
  - TD3:  https://arxiv.org/abs/1802.09477 (twin critics + delayed policy updates for DDPG)
"""

from algos.matd3.policy import MATD3Policy
from algos.matd3.trainer import MATD3Trainer

__all__ = ["MATD3Policy", "MATD3Trainer"]

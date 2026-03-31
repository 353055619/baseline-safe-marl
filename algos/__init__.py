"""
algos/__init__.py — MARL Algorithm Stub Registry
================================================
Exposes BasePolicy, BaseTrainer, RolloutBuffer, and all algorithm stubs.
"""

from algos.base import BasePolicy, BaseTrainer, RolloutBuffer
from algos.facmac import FACMACPolicy, FACMACTrainer

__all__ = [
    "BasePolicy",
    "BaseTrainer",
    "RolloutBuffer",
    "FACMACPolicy",
    "FACMACTrainer",
]

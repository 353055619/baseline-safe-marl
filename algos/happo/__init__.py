"""
algs/happo/__init__.py — HAPPO Algorithm Stub
==============================================
HAPPO: Multi-Agent HPPO (Heterogeneous-agent Proximal Policy Optimization).

Exports:
    HAPPOPolicy
    HAPPOTrainer
"""

from algos.happo.policy import HAPPOPolicy
from algos.happo.trainer import HAPPOTrainer

__all__ = ["HAPPOPolicy", "HAPPOTrainer"]

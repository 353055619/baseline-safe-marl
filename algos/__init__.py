"""
algos/__init__.py — MARL Algorithm Stub Registry
================================================
Exposes BasePolicy, BaseTrainer, RolloutBuffer, and all algorithm stubs.
"""

from algos.base import BasePolicy, BaseTrainer, RolloutBuffer

from algos.facmac import FACMACPolicy, FACMACTrainer
from algos.happo import HAPPOPolicy, HAPPOTrainer
from algos.macpo import MACPOPolicy, MACPOTrainer
from algos.mappo import MAPPOPolicy, MAPPOTrainer
from algos.mappo_lagrangian import MAPPOLPolicy, MAPPOLTrainer
from algos.matd3 import MATD3Policy, MATD3Trainer

__all__ = [
    # bases
    "BasePolicy",
    "BaseTrainer",
    "RolloutBuffer",
    # stubs
    "FACMACPolicy",
    "FACMACTrainer",
    "HAPPOPolicy",
    "HAPPOTrainer",
    "MACPOPolicy",
    "MACPOTrainer",
    "MAPPOPolicy",
    "MAPPOTrainer",
    "MAPPOLPolicy",
    "MAPPOLTrainer",
    "MATD3Policy",
    "MATD3Trainer",
]

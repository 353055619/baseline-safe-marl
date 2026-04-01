"""
baseline_safe_marl.algos.registry — Algorithm Registry
======================================================
"""
from __future__ import annotations
from typing import Tuple, Type

from baseline_safe_marl.algos.base import BasePolicy, BaseTrainer


ALGO_REGISTRY: dict[str, Tuple[str, str]] = {
    "MAPPO": (
        "baseline_safe_marl.algos.on_policy.mappo:MAPPOPolicy",
        "baseline_safe_marl.algos.on_policy.mappo:MAPPOTrainer",
    ),
    "MAPPO-L": (
        "baseline_safe_marl.algos.on_policy.mappo_lagrangian:MAPPOLPolicy",
        "baseline_safe_marl.algos.on_policy.mappo_lagrangian:MAPPOLTrainer",
    ),
    "HAPPO": (
        "baseline_safe_marl.algos.on_policy.happo:HAPPOPolicy",
        "baseline_safe_marl.algos.on_policy.happo:HAPPOTrainer",
    ),
    "MACPO": (
        "baseline_safe_marl.algos.on_policy.macpo:MACPOPolicy",
        "baseline_safe_marl.algos.on_policy.macpo:MACPOTrainer",
    ),
    "MATD3": (
        "baseline_safe_marl.algos.off_policy.matd3:MATD3Policy",
        "baseline_safe_marl.algos.off_policy.matd3:MATD3Trainer",
    ),
    "FACMAC": (
        "baseline_safe_marl.algos.off_policy.facmac:FACMACPolicy",
        "baseline_safe_marl.algos.off_policy.facmac:FACMACTrainer",
    ),
}


def get_algo(name: str) -> Tuple[Type[BasePolicy], Type[BaseTrainer]]:
    if name not in ALGO_REGISTRY:
        raise KeyError(f"Algorithm '{name}' not found. Available: {list(ALGO_REGISTRY.keys())}")

    policy_path, trainer_path = ALGO_REGISTRY[name]

    policy_mod_name, policy_cls_name = policy_path.split(":")
    trainer_mod_name, trainer_cls_name = trainer_path.split(":")

    policy_mod = __import__(policy_mod_name, fromlist=[policy_cls_name])
    trainer_mod = __import__(trainer_mod_name, fromlist=[trainer_cls_name])

    return getattr(policy_mod, policy_cls_name), getattr(trainer_mod, trainer_cls_name)


def list_algos() -> list[str]:
    return list(ALGO_REGISTRY.keys())

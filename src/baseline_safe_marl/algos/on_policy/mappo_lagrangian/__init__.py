"""MAPPO-L policy and trainer"""
from baseline_safe_marl.algos.on_policy.mappo_lagrangian.policy import MAPPOLPolicy
from baseline_safe_marl.algos.on_policy.mappo_lagrangian.trainer import MAPPOLTrainer
__all__ = ["MAPPOLPolicy", "MAPPOLTrainer"]

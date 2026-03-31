"""
algos/registry.py — Algorithm Registry
======================================
算法名到 (PolicyClass, TrainerClass) 的映射，供 runner/launcher 使用。

用法：
    from algos.registry import ALGO_REGISTRY, get_algo

    PolicyCls, TrainerCls = ALGO_REGISTRY["MATD3"]
    policy = PolicyCls(cfg)
    trainer = TrainerCls(cfg, policy)

当前覆盖算法（5 个）：
    MAPPO         — on-policy PPO, shared actor-critic
    MAPPO-L       — MAPPO + Lagrangian cost constraint
    HAPPO         — on-policy, heterogeneous agents, trust region
    MACPO         — on-policy, MACPO constraint (CVPO-style)
    MATD3         — off-policy DDPG family, twin critics, delayed policy update
"""

from __future__ import annotations

from typing import Tuple

from algos.base import BasePolicy, BaseTrainer


# --------------------------------------------------------------------
# Registry mapping
# --------------------------------------------------------------------
ALGO_REGISTRY: dict[str, Tuple[str, str]] = {
    "MAPPO": (
        "algos.mappo:MAPPOPolicy",
        "algos.mappo:MAPPOTrainer",
    ),
    "MAPPO-L": (
        "algos.mappo_lagrangian:MAPPOLPolicy",
        "algos.mappo_lagrangian:MAPPOLTrainer",
    ),
    "HAPPO": (
        "algos.happo:HAPPOPolicy",
        "algos.happo:HAPPOTrainer",
    ),
    "MACPO": (
        "algos.macpo:MACPOPolicy",
        "algos.macpo:MACPOTrainer",
    ),
    "MATD3": (
        "algos.matd3:MATD3Policy",
        "algos.matd3:MATD3Trainer",
    ),
}


def get_algo(name: str) -> Tuple[type[BasePolicy], type[BaseTrainer]]:
    """
    解析并返回算法对应的 Policy 和 Trainer 类。

    Parameters
    ----------
    name : str
        算法名，如 "MATD3", "MAPPO", "HAPPO"

    Returns
    -------
    Tuple[type[BasePolicy], type[BaseTrainer]]

    Raises
    ------
    KeyError
        如果算法未在 ALGO_REGISTRY 中注册。
    ImportError
        如果算法模块无法导入。
    """
    if name not in ALGO_REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not found in registry. "
            f"Available: {list(ALGO_REGISTRY.keys())}"
        )

    policy_path, trainer_path = ALGO_REGISTRY[name]

    policy_mod_name, policy_cls_name = policy_path.split(":")
    trainer_mod_name, trainer_cls_name = trainer_path.split(":")

    policy_mod = __import__(policy_mod_name, fromlist=[policy_cls_name])
    trainer_mod = __import__(trainer_mod_name, fromlist=[trainer_cls_name])

    PolicyCls = getattr(policy_mod, policy_cls_name)
    TrainerCls = getattr(trainer_mod, trainer_cls_name)

    return PolicyCls, TrainerCls


# --------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------
def list_algos() -> list[str]:
    """返回所有已注册算法名。"""
    return list(ALGO_REGISTRY.keys())

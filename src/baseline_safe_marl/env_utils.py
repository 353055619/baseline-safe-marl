"""
环境创建与评估工具
"""
from typing import Dict, Any
from baseline_safe_marl.envs import make_safe_ant_2x4, make_safe_halfcheetah_2x3, make_safe_hopper_2


_ENV_FACTORIES = {
    "safeant2x4": make_safe_ant_2x4,
    "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
    "safehopper2": make_safe_hopper_2,
}


def make_env(env_name: str, **kwargs):
    factory = _ENV_FACTORIES.get(env_name.lower())
    if factory is None:
        raise ValueError(f"Unknown env: {env_name}. Available: {list(_ENV_FACTORIES.keys())}")
    return factory(**kwargs)


def get_env_names():
    return list(_ENV_FACTORIES.keys())

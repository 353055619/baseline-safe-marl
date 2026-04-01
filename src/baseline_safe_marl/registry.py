"""
算法和环境注册表
"""
from typing import Dict, Type, Callable, Any


_ALGO_POLICY_REGISTRY: Dict[str, Type] = {}
_ALGO_TRAINER_REGISTRY: Dict[str, Type] = {}
_ENV_FACTORY_REGISTRY: Dict[str, Callable] = {}


def register_algo_policy(name: str):
    """装饰器：注册策略类"""
    def deco(cls):
        _ALGO_POLICY_REGISTRY[name.upper()] = cls
        return cls
    return deco


def register_algo_trainer(name: str):
    """装饰器：注册训练器类"""
    def deco(cls):
        _ALGO_TRAINER_REGISTRY[name.upper()] = cls
        return cls
    return deco


def register_env(name: str):
    """装饰器：注册环境工厂"""
    def deco(fn):
        _ENV_FACTORY_REGISTRY[name.lower()] = fn
        return fn
    return deco


def get_policy(name: str) -> Type:
    return _ALGO_POLICY_REGISTRY[name.upper()]


def get_trainer(name: str) -> Type:
    return _ALGO_TRAINER_REGISTRY[name.upper()]


def get_env(name: str) -> Callable:
    return _ENV_FACTORY_REGISTRY[name.lower()]


def list_algos() -> list:
    return list(_ALGO_POLICY_REGISTRY.keys())

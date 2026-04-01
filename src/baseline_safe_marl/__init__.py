"""
baseline_safe_marl — Safe Multi-Agent Reinforcement Learning Baseline
"""
__version__ = "0.1.0"

from baseline_safe_marl.config import load_config
from baseline_safe_marl.algo_config import make_algo_config

__all__ = [
    "load_config",
    "make_algo_config",
]

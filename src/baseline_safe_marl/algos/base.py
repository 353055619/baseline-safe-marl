"""
algos/base.py — MARL Algorithm Base Classes
============================================
定义所有算法 stub 必须实现的抽象接口。

所有具体算法（MAPPO, MAPPO-L 等）都必须继承这些基类。

设计原则：
- 环境与算法通过接口解耦
- Trainer 和 Policy 分离，方便独立测试
- Stub 阶段只要求能 instantiate，不要求真实 forward/backward
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------
# Policy Base
# --------------------------------------------------------------------
class BasePolicy(ABC, torch.nn.Module):
    """
    所有策略网络的基类。

    子类必须实现：
        get_actions(obs, deterministic)
        evaluate_actions(obs, actions)
        update(rollout_buffer) -> dict
    """

    def __init__(self, cfg: Dict[str, Any], agent_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = cfg.get("device", "cpu")

    @abstractmethod
    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        根据 obs 返回 action。

        参数:
            obs: 当前 observation (numpy array)
            deterministic: True = 贪心策略，False = 随机策略

        返回:
            action: numpy array
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Dict[str, Any]:
        """
        评估 (obs, actions) 对的 log_prob / entropy 等。

        返回:
            dict with keys: log_prob, entropy, value (optional)
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """保存策略权重到文件"""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """从文件加载策略权重"""
        self.load_state_dict(torch.load(path, map_location=self.device))


# --------------------------------------------------------------------
# Trainer Base
# --------------------------------------------------------------------
class BaseTrainer(ABC):
    """
    所有训练器的基类。

    子类必须实现：
        train()
        update_lagrangian()  [仅约束优化算法]
        save(path) / load(path)
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        self.cfg = cfg
        self.policy = policy
        self.device = cfg.get("device", "cpu")
        self.total_steps = 0

    @abstractmethod
    def train(self, num_steps: int) -> Dict[str, float]:
        """
        执行一次训练 step。

        参数:
            num_steps: 本次训练执行的步数

        返回:
            dict: 训练指标（如 loss, kl, entropy 等）
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """保存 trainer 状态到文件"""
        state = {
            "policy": self.policy.state_dict(),
            "total_steps": self.total_steps,
            "cfg": self.cfg,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """从文件加载 trainer 状态"""
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.total_steps = state.get("total_steps", 0)


# --------------------------------------------------------------------
# Rollout Buffer Base（stub，只定义接口，不要求完整实现）
# --------------------------------------------------------------------
class RolloutBuffer:
    """
    存放一个 rollout 轨迹的数据结构。

    Stub 阶段：只需要能存储和按序读取，不需要完整实现。
    """

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.obs)

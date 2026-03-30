"""
safe_mamujoco_adapter.py — Phase 1 Fallback Adapter
=====================================================
⚠️ 这是 fallback 原型验证路线，不是最终 benchmark。
结果仅用于验证集成链路，不作为安全性能对比依据。

接口层：对外暴露 PettingZoo-style Parallel API，与原生 Safe MAMujoco 接口对齐。
内部实现：用 gymnasium 单智能体环境 + 逻辑多智能体 wrapper，
         不依赖 Safe MAMujoco / mujoco_py / PettingZoo mujoco。

目标接口（与原生 Safe MAMujoco 对齐）：
    make_safe_ant_2x4()       -> PettingZoo ParallelEnv
    make_safe_halfcheetah_2x3()
    make_safe_hopper_2()
    SafeMAMujocoAdapter       基类

用法示例：
    from envs.safe_mamujoco_adapter import make_safe_ant_2x4
    env = make_safe_ant_2x4()
    obs_dict, info_dict = env.reset(seed=42)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs_dict, rewards, terms, truncs, info_dict = env.step(actions)
    env.close()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from envs.fallback_cost_wrapper import (
    CostWrapper,
    MultiAgentCostWrapper,
    make_ant_fall_cost,
    make_halfcheetah_cost,
    make_hopper_cost,
)


# --------------------------------------------------------------------
# 类型别名
# --------------------------------------------------------------------
ObsDict = Dict[str, np.ndarray]
ActDict = Dict[str, np.ndarray]
RewDict = Dict[str, float]
TermDict = Dict[str, bool]
TruncDict = Dict[str, bool]
InfoDict = Dict[str, Any]
CostDict = Dict[str, float]


# --------------------------------------------------------------------
# 逻辑多智能体基础类（fake multi-agent，共享同一个单智能体 gymnasium env）
# --------------------------------------------------------------------
class MAMujocoFakeAdapter:
    """
    将 gymnasium 单智能体环境包装为逻辑上的多智能体环境。

    实现逻辑：
    - 每个"智能体"是逻辑上的 agent，不独立拥有物理 body
    - 所有 agent 共享同一个 gymnasium env 的 state
    - 动作空间是原始 env.action_space 的均匀切分
    - 观察空间是原始 env.observation_space 的均匀切分

    ⚠️ 这是 fallback stub，与真实 multi-agent physics 完全不同。

    PettingZoo ParallelEnv API：
        .agents: List[str]
        .action_space(agent): gym.Space
        .observation_space(agent): gym.Space
        .reset(seed=None) -> (obs_dict, info_dict)
        .step(actions: act_dict) -> (obs, rew, term, trunc, info)
        .close()
    """

    # 子类覆盖
    _AGENTS: List[str] = []

    def __init__(
        self,
        env: gym.Env,
        n_agents: int,
        action_slices: Optional[List[slice]] = None,
        obs_slices: Optional[List[slice]] = None,
        cost_wrapper_class=CostWrapper,
        cost_fn=None,
        render_mode: str = "rgb_array",
    ):
        self.n_agents = n_agents
        self._env = env
        self._render_mode = render_mode

        # 动作/观察空间切片
        self._action_slices = action_slices or self._default_action_slices(n_agents)
        self._obs_slices = obs_slices or self._default_obs_slices(n_agents)

        # 包装 cost
        self._cost_wrapper_class = cost_wrapper_class
        self._cost_fn = cost_fn
        if cost_wrapper_class and cost_fn:
            self._env = cost_wrapper_class(self._env, cost_fn=cost_fn)

        self._agents = [f"agent_{i}" for i in range(n_agents)]

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv API
    # ------------------------------------------------------------------
    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def num_agents(self) -> int:
        return self.n_agents

    def action_space(self, agent: str) -> gym.Space:
        """返回该 agent 对应的动作切片空间（全空间共享，无感知）"""
        return self._env.action_space

    def observation_space(self, agent: str) -> gym.Space:
        return self._env.observation_space

    def reset(self, seed: Optional[int] = None) -> Tuple[ObsDict, InfoDict]:
        obs, info = self._env.reset(seed=seed)
        # 注入 zero cost
        info["cost"] = {a: 0.0 for a in self.agents}
        obs_dict = {a: self._slice_obs(obs) for a in self.agents}
        return obs_dict, info

    def step(
        self, actions: ActDict
    ) -> Tuple[ObsDict, RewDict, TermDict, TruncDict, InfoDict]:
        """
        将多个 agent 的动作拼接成原始 env 的动作，执行一步。
        """
        # 简单策略：各 agent 动作取平均后执行
        action = np.zeros(self._env.action_space.shape, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            if agent in actions:
                a = np.asarray(actions[agent], dtype=np.float32)
                if a.shape == action.shape:
                    action += a
                elif a.ndim == 0:
                    action += a.item()
        action = action / max(1, len(actions))

        obs, reward, terminated, truncated, info = self._env.step(action)

        obs_dict = {a: self._slice_obs(obs) for a in self.agents}
        rewards = {a: float(reward) for a in self.agents}
        terms = {a: bool(terminated) for a in self.agents}
        truncs = {a: bool(truncated) for a in self.agents}

        # cost 注入
        cost = info.get("cost", 0.0)
        info["cost"] = {a: cost for a in self.agents}

        return obs_dict, rewards, terms, truncs, info

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _slice_obs(self, obs: np.ndarray) -> np.ndarray:
        """将完整 obs 切片给各 agent（stub 版直接返回完整 obs）"""
        return np.asarray(obs, dtype=np.float32)

    def _default_action_slices(self, n: int) -> List[slice]:
        total = self._env.action_space.shape[0]
        chunk = total // n
        return [slice(i * chunk, (i + 1) * chunk) for i in range(n)]

    def _default_obs_slices(self, n: int) -> List[slice]:
        total = self._env.observation_space.shape[0]
        chunk = total // n
        return [slice(i * chunk, (i + 1) * chunk) for i in range(n)]


# --------------------------------------------------------------------
# 环境工厂函数（与原生 Safe MAMujoco 接口签名一致）
# --------------------------------------------------------------------
def make_safe_ant_2x4(
    render_mode: str = "rgb_array",
) -> MAMujocoFakeAdapter:
    """
    Safe Ant-2x4 fallback.

    2 agents 逻辑上各控制 4 个关节（stub）。
    Cost: 摔倒（torso_z < 0.25）或偏离走廊（y > 5 或 y < -5）。

    等价原生接口：schroederdewitt/multiagent_mujoco Ant-2x4
    """
    base = gym.make("Ant-v5", render_mode=render_mode)
    return MAMujocoFakeAdapter(
        env=base,
        n_agents=2,
        cost_wrapper_class=CostWrapper,
        cost_fn=make_ant_fall_cost(),
        render_mode=render_mode,
    )


def make_safe_halfcheetah_2x3(
    render_mode: str = "rgb_array",
) -> MAMujocoFakeAdapter:
    """
    Safe HalfCheetah-2x3 fallback.

    2 agents 逻辑上各控制 3 个关节（stub）。
    Cost: 超速（velocity_x > 3.0 m/s）。

    等价原生接口：schroederdewitt/multiagent_mujoco HalfCheetah-2x3
    """
    base = gym.make("HalfCheetah-v5", render_mode=render_mode)
    return MAMujocoFakeAdapter(
        env=base,
        n_agents=2,
        cost_wrapper_class=CostWrapper,
        cost_fn=make_halfcheetah_cost(speed_threshold=3.0),
        render_mode=render_mode,
    )


def make_safe_hopper_2(
    render_mode: str = "rgb_array",
) -> MAMujocoFakeAdapter:
    """
    Safe Hopper-2 fallback.

    2 agents 逻辑上各控制 hopper 的一半（stub）。
    Cost: 不稳定（动作幅度过大作为代理）。

    等价原生接口：schroederdewitt/multiagent_mujoco Hopper-2
    """
    base = gym.make("Hopper-v5", render_mode=render_mode)
    return MAMujocoFakeAdapter(
        env=base,
        n_agents=2,
        cost_wrapper_class=CostWrapper,
        cost_fn=make_hopper_cost(),
        render_mode=render_mode,
    )


def make_safe_walker_2(
    render_mode: str = "rgb_array",
) -> MAMujocoFakeAdapter:
    """
    Safe Walker-2 fallback.

    2 agents 逻辑上各控制 walker 的一半（stub）。
    Cost: 摔倒（torso_z < 0.3 proxy）。

    等价原生接口：schroederdewitt/multiagent_mujoco Walker-2d-2
    """
    base = gym.make("Walker2d-v5", render_mode=render_mode)

    def walker_cost_fn(env, obs, action):
        # torso_z 在 walker 约 obs[0]（简化 proxy）
        return 1.0 if len(obs) > 0 and obs[0] < 0.3 else 0.0

    return MAMujocoFakeAdapter(
        env=base,
        n_agents=2,
        cost_wrapper_class=CostWrapper,
        cost_fn=walker_cost_fn,
        render_mode=render_mode,
    )


# --------------------------------------------------------------------
# PettingZoo ParallelEnv 注册（用于 MAPPO runner）
# --------------------------------------------------------------------
def to_pettingzoo_env(mamujoco_adapter: MAMujocoFakeAdapter) -> MAMujocoFakeAdapter:
    """
    透传：fallback adapter 已实现 PettingZoo-style 接口，
    此函数作为未来切换到真实 PettingZoo 环境时的统一入口。
    未来替换时，只需改动这里：
        return pettingzoo.mujoco.ant_v3.parallel_env(...)
    而算法代码（MAPPO/MAPPO-L）无需改动。
    """
    return mamujoco_adapter


# --------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("=== safe_mamujoco_adapter smoke test ===")
    print("⚠️  fallback 原型验证，非 benchmark")

    test_cases = [
        ("Ant-2x4", make_safe_ant_2x4),
        ("HalfCheetah-2x3", make_safe_halfcheetah_2x3),
        ("Hopper-2", make_safe_hopper_2),
        ("Walker-2", make_safe_walker_2),
    ]

    for name, factory in test_cases:
        try:
            env = factory()
            assert hasattr(env, "agents"), f"{name}: missing .agents"
            assert hasattr(env, "reset"), f"{name}: missing .reset"
            assert hasattr(env, "step"), f"{name}: missing .step"

            obs_dict, info_dict = env.reset(seed=42)
            for a in env.agents:
                assert a in obs_dict, f"{name}: {a} not in obs_dict"
                assert "cost" in info_dict, f"{name}: cost missing from info"
                assert env.action_space(a) is not None, f"{name}: action_space None"

            for i in range(50):
                actions = {a: env.action_space(a).sample() for a in env.agents}
                obs_dict, rewards, terms, truncs, info_dict = env.step(actions)
                for a in env.agents:
                    assert a in obs_dict, f"{name} step {i}: {a} obs missing"
                    assert "cost" in info_dict, f"{name} step {i}: cost missing"
                    assert info_dict["cost"].get(a) in (0.0, 1.0), f"{name}: bad cost value"

            env.close()
            print(f"[PASS] {name}: 50 steps OK")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback; traceback.print_exc()

    print("=== smoke test complete ===")

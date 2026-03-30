"""
Fallback CostWrapper for baseline-safe-marl
==========================================
⚠️ 这是 fallback 原型验证路线，不是最终 benchmark。

在任意 gymnasium 环境上叠加 cost 信号，模拟 Safe MAMujoco 的安全约束。
Cost 规则：
  - HalfCheetah: cost=1 if abs(velocity_x) > 3.0 m/s（超速惩罚）
  - Ant: cost=1 if torso_z < 0.25（摔倒）or torso_y > 5.0 or torso_y < -5.0（越界）
  - 其他/默认: cost=0.0（暂不施加约束，仅保留接口）

用法示例：
  from envs.fallback_cost_wrapper import CostWrapper, make_halfcheetah_cost
  from envs.fallback_cost_wrapper import make_ant_fall_cost  # Ant 摔倒 cost

  env = gym.make("HalfCheetah-v5")
  env = CostWrapper(env, cost_fn=make_halfcheetah_cost())
  obs, info = env.reset(seed=42)      # info["cost"] == 0.0
  obs, r, term, trunc, info = env.step(action)  # info["cost"] ∈ {0.0, 1.0}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


# --------------------------------------------------------------------
# Cost Function Types
# --------------------------------------------------------------------
# Single-agent cost function: (env, obs, action) -> float
CostFnSingle = Callable[[gym.Env, np.ndarray, np.ndarray], float]

# Multi-agent cost function: (env, obs_dict, action_dict) -> Dict[agent_name, float]
CostFnMulti = Callable[
    [gym.Env, Dict[str, np.ndarray], Dict[str, np.ndarray]], Dict[str, float]
]

CostFn = Union[CostFnSingle, CostFnMulti, None]


# --------------------------------------------------------------------
# Fallback Cost Function Presets
# --------------------------------------------------------------------

def make_halfcheetah_cost(speed_threshold: float = 3.0) -> CostFnSingle:
    """
    HalfCheetah cost: penalise running too fast.
    Safe MAMujoco reference: the original uses wall proximity;
    here we use a velocity proxy to demonstrate the cost interface.

    HalfCheetah-v5 obs layout (17 dims):
        qpos[0:8] + qvel[8:17]
    x-velocity is qvel[1], i.e. obs[9].
    """
    def cost_fn(env: gym.Env, obs: np.ndarray, action: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float64)
        nq = 8   # number of qpos entries for HalfCheetah-v5
        xvel = float(np.abs(obs[nq + 1])) if len(obs) > nq + 1 else 0.0
        return 1.0 if xvel > speed_threshold else 0.0
    return cost_fn


def make_ant_fall_cost(
    torso_z_threshold: float = 0.25,
    y_boundary: float = 5.0,
) -> CostFnSingle:
    """
    Ant cost: penalise falling (torso too low) or leaving corridor.
    Mirrors Safe MAMujoco Ant task: walls at y=±5, floor cost if torso_z < 0.25.
    """
    def cost_fn(env: gym.Env, obs: np.ndarray, action: np.ndarray) -> float:
        # Ant qpos layout: [x, y, z, quat(4)] = first 5 dims
        # torso z is obs[2]; torso y is obs[1]
        if len(obs) >= 3:
            torso_z = float(obs[2])
            torso_y = float(obs[1]) if len(obs) >= 2 else 0.0
        else:
            torso_z, torso_y = 0.0, 0.0

        if torso_z < torso_z_threshold:
            return 1.0   # fell over
        if abs(torso_y) > y_boundary:
            return 1.0   # left safe corridor
        return 0.0
    return cost_fn


def make_hopper_cost(contact_threshold: float = 1.0) -> CostFnSingle:
    """
    Hopper cost: penalise excessive contact force (as proxy for instability).
    """
    def cost_fn(env: gym.Env, obs: np.ndarray, action: np.ndarray) -> float:
        # Hopper obs: qpos(6) + qvel(6) = 12 dims; contact forces unavailable
        # directly in obs. Use action magnitude as instability proxy.
        action_mag = float(np.linalg.norm(action))
        return 1.0 if action_mag > contact_threshold * 10 else 0.0
    return cost_fn


def zero_cost(env: gym.Env, obs: np.ndarray, action: np.ndarray) -> float:
    """Default no-op cost function."""
    return 0.0


# --------------------------------------------------------------------
# CostWrapper (single-agent gymnasium Env)
# --------------------------------------------------------------------
class CostWrapper(gym.Wrapper):
    """
    Wraps a gymnasium environment and injects a ``cost`` field into
    the ``info`` dict returned by every ``reset()`` / ``step()`` call.

    Signature parity with Safe MAMujoco:
        reset(seed=None)  -> (obs, info)     where info["cost"] == 0.0
        step(action)      -> (obs, r, term, trunc, info)
                             info["cost"] == 0.0 or 1.0

    Parameters
    ----------
    env : gym.Env
        The underlying gymnasium environment to wrap.
    cost_fn : callable, optional
        Signature ``cost = cost_fn(env, obs, action) -> float``.
        If None, defaults to zero-cost (no constraint).

    Examples
    --------
    >>> from envs.fallback_cost_wrapper import CostWrapper, make_halfcheetah_cost
    >>> import gymnasium as gym
    >>> env = CostWrapper(gym.make("HalfCheetah-v5"), cost_fn=make_halfcheetah_cost())
    >>> obs, info = env.reset(seed=42)
    >>> assert "cost" in info and info["cost"] == 0.0
    >>> obs, r, term, trunc, info = env.step(env.action_space.sample())
    >>> assert "cost" in info
    >>> env.close()
    """

    def __init__(
        self,
        env: gym.Env,
        cost_fn: Optional[CostFnSingle] = None,
    ):
        super().__init__(env)
        self._cost_fn = cost_fn if cost_fn is not None else zero_cost

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        info["cost"] = 0.0
        return obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["cost"] = self._cost_fn(self.env, obs, action)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def cost_fn(self) -> CostFnSingle:
        """Return the current cost function."""
        return self._cost_fn


# --------------------------------------------------------------------
# MultiAgentCostWrapper (PettingZoo parallel / AEC env)
# --------------------------------------------------------------------
class MultiAgentCostWrapper:
    """
    Lightweight adapter that wraps a PettingZoo parallel_env or AEC_env
    and adds a per-agent ``cost`` key to each info dict.

    Designed for use with MAPPO / HAPPO style training loops that
    consume PettingZoo environments.

    Parameters
    ----------
    env : PettingZoo env (parallel_env or AECEnv)
    cost_fn : callable, optional
        Signature ``cost_dict = cost_fn(env, obs_dict, action_dict) -> Dict[agent, float]``

    Examples
    --------
    >>> from envs.fallback_cost_wrapper import MultiAgentCostWrapper, make_ant_fall_cost
    >>> from pettingzoo.mujoco import hopper_v3
    >>> base_env = hopper_v3.parallel_env(render_mode="rgb_array")
    >>> env = MultiAgentCostWrapper(base_env, cost_fn=make_ant_fall_cost())
    >>> obs_dict, info_dict = env.reset(seed=42)
    >>> actions = {a: env.action_space(a).sample() for a in env.agents}
    >>> obs_dict, rewards, terms, truncs, info_dict = env.step(actions)
    >>> assert "cost" in info_dict[env.agents[0]]
    >>> env.close()
    """

    def __init__(
        self,
        env,  # PettingZoo env, no gymnasium type hint to avoid hard dep
        cost_fn: Optional[CostFnMulti] = None,
    ):
        self._env = env
        self._cost_fn = cost_fn if cost_fn is not None else self._zero_multi_cost

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        obs_dict, info_dict = self._env.reset(seed=seed)
        # inject zero cost on reset
        for k in obs_dict:
            info_dict.setdefault("cost", {})[k] = 0.0
        return obs_dict, info_dict

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        obs_dict, rewards, terms, truncs, info_dict = self._env.step(actions)
        # compute per-agent costs
        costs = self._cost_fn(self._env, obs_dict, actions)
        for agent in self._env.agents:
            info_dict.setdefault("cost", {})[agent] = costs.get(agent, 0.0)
        return obs_dict, rewards, terms, truncs, info_dict

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str):
        """Proxy attribute access to the underlying env."""
        return getattr(self._env, name)

    @staticmethod
    def _zero_multi_cost(env, obs_dict, action_dict) -> Dict[str, float]:
        return {agent: 0.0 for agent in env.agents}


# --------------------------------------------------------------------
# Convenience factory functions
# --------------------------------------------------------------------
def make_halfcheetah_cost_wrapper(
    render_mode: str = "rgb_array",
    speed_threshold: float = 3.0,
) -> CostWrapper:
    """
    Factory: HalfCheetah-v5 + CostWrapper with velocity-based cost.
    """
    env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    return CostWrapper(env, cost_fn=make_halfcheetah_cost(speed_threshold))


def make_ant_cost_wrapper(
    render_mode: str = "rgb_array",
) -> CostWrapper:
    """
    Factory: Ant-v5 + CostWrapper with fall/or-out cost.
    """
    env = gym.make("Ant-v5", render_mode=render_mode)
    return CostWrapper(env, cost_fn=make_ant_fall_cost())


# --------------------------------------------------------------------
# Smoke test (runs when executed directly)
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("=== CostWrapper smoke test ===")

    # -- single-agent HalfCheetah --
    try:
        env = make_halfcheetah_cost_wrapper(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert "cost" in info, "reset: cost missing from info"
        assert info["cost"] == 0.0, "reset: initial cost should be 0.0"

        for i in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert "cost" in info, f"step {i}: cost missing from info"
            assert info["cost"] in (0.0, 1.0), f"step {i}: cost must be 0.0 or 1.0, got {info['cost']}"
            if term or trunc:
                obs, info = env.reset(seed=42)

        env.close()
        print("[PASS] HalfCheetah + CostWrapper: 100 steps OK")
    except Exception as e:
        print(f"[FAIL] HalfCheetah + CostWrapper: {e}")
        import traceback; traceback.print_exc()

    # -- single-agent Ant --
    try:
        env = make_ant_cost_wrapper(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert "cost" in info and info["cost"] == 0.0

        for i in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert "cost" in info and info["cost"] in (0.0, 1.0)
            if term or trunc:
                obs, info = env.reset(seed=42)

        env.close()
        print("[PASS] Ant + CostWrapper: 100 steps OK")
    except Exception as e:
        print(f"[FAIL] Ant + CostWrapper: {e}")
        import traceback; traceback.print_exc()

    print("=== smoke test complete ===")

"""
scripts/demo_episode_mappo.py — Minimal End-to-End Episode Demo
===============================================================
目标：验证 MAPPO stub + fallback CostWrapper + safe_mamujoco_adapter
      能完整跑通 1 episode（reset → 多 step → episode done）。

⚠️ 这是集成验证 demo，不是训练脚本。随机策略，不做学习。

配置系统接入：
  1. load_config() 加载 configs/phase1_default.yaml
  2. make_algo_config("MAPPO", cfg) 注入 algo-specific 默认字段
  3. env 从 cfg["env"]["env_name"] + cfg["env"]["fallback"] 选择

Usage:
    cd /path/to/baseline-safe-marl
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode_mappo.py
    # 或用 CLI override:
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode_mappo.py --config.env.env_name=SafeHalfcheetah2x3
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Config system
from src.config import load_config
from src.algo_config import make_algo_config

# Env
from envs.safe_mamujoco_adapter import (
    make_safe_ant_2x4,
    make_safe_halfcheetah_2x3,
    make_safe_hopper_2,
)
from algos.mappo import MAPPOPolicy


# --------------------------------------------------------------------
# Env factory map (fallback adapters)
# --------------------------------------------------------------------
_FALLBACK_ENV_FACTORIES = {
    "safeant2x4": make_safe_ant_2x4,
    "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
    "safehopper2": make_safe_hopper_2,
}


def run_episode(policy: MAPPOPolicy, env, max_steps: int = 200) -> dict:
    """Run one episode. Returns collected stats."""
    obs_dict, info_dict = env.reset(seed=42)
    total_reward = {agent: 0.0 for agent in env.agents}
    total_cost = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    print(f"\nEpisode start: {len(env.agents)} agents, max_steps={max_steps}")
    print(f"  obs keys: {list(obs_dict.keys())}")

    while not done and step_count < max_steps:
        action_dict = {}
        for agent in env.agents:
            action_dict[agent] = policy.get_actions(obs_dict[agent], deterministic=True)

        obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)

        for agent in env.agents:
            total_reward[agent] += rewards.get(agent, 0.0)
            total_cost[agent] += info_dict.get("cost", {}).get(agent, 0.0)

        step_count += 1
        done = all(terms.values()) or all(truncs.values())

        if step_count % 50 == 0 or done:
            r_vals = [f"{total_reward[a]:.1f}" for a in list(env.agents)[:2]]
            c_vals = [f"{total_cost[a]:.1f}" for a in list(env.agents)[:2]]
            print(f"  step {step_count:3d}: done={done} "
                  f"rewards={r_vals} costs={c_vals}")

    return {
        "steps": step_count,
        "total_reward": total_reward,
        "total_cost": total_cost,
        "done": done,
    }


def main():
    print("=" * 60)
    print("MAPPO + Fallback Safe MAMujoco — End-to-End Episode Demo")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # ------------------------------------------------------------------
    # 1. Load config from YAML + CLI overrides
    # ------------------------------------------------------------------
    print("\n[1] Loading config...")
    try:
        base_cfg = load_config(config_path="configs/phase1_default.yaml")
    except FileNotFoundError:
        print("  WARN: configs/phase1_default.yaml not found, using hardcoded fallback cfg")
        base_cfg = {
            "algo": {"algo_name": "MAPPO", "hidden_dim": 64, "activation": "tanh", "lr": 3e-4},
            "env": {"env_name": "safehalfcheetah2x3", "fallback": True, "render_mode": None},
        }
    cfg = make_algo_config(base_cfg.get("algo", {}).get("algo_name", "MAPPO"), base_cfg)

    env_name_raw = cfg.get("env", {}).get("env_name", "safeant2x4").lower()
    render_mode = cfg.get("env", {}).get("render_mode", None)
    algo_name = cfg.get("algo", {}).get("algo_name", "MAPPO")
    hidden_dim = cfg.get("algo", {}).get("hidden_dim", 64)
    lr = cfg.get("algo", {}).get("lr", 3e-4)

    print(f"   algo={algo_name}  env={env_name_raw}  hidden_dim={hidden_dim}  lr={lr}")

    # ------------------------------------------------------------------
    # 2. Create env (fallback adapter)
    # ------------------------------------------------------------------
    print("\n[2] Creating fallback env...")
    factory = _FALLBACK_ENV_FACTORIES.get(env_name_raw)
    if factory is None:
        print(f"  WARN: unknown env '{env_name_raw}', falling back to safehalfcheetah2x3")
        factory = make_safe_halfcheetah_2x3

    try:
        env = factory(render_mode=render_mode)
        print("   env created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    # ------------------------------------------------------------------
    # 3. Auto-detect obs/action dim from env
    # ------------------------------------------------------------------
    obs_sample, _ = env.reset(seed=42)
    sample_agent = env.agents[0]
    obs_dim = int(np.array(obs_sample[sample_agent]).shape[-1])
    action_dim = int(np.array(env.action_space(sample_agent).sample()).shape[-1])
    env.close()

    cfg["env"]["obs_dim"] = obs_dim
    cfg["env"]["action_dim"] = action_dim
    print(f"   obs_dim={obs_dim}, action_dim={action_dim}")

    # ------------------------------------------------------------------
    # 4. Create MAPPO policy (using enriched config)
    # ------------------------------------------------------------------
    print("\n[3] Creating MAPPOPolicy from enriched config...")
    try:
        policy = MAPPOPolicy(cfg, agent_id=0)
        print("   MAPPOPolicy created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 5. Run episode
    # ------------------------------------------------------------------
    print("\n[4] Running 1 episode...")
    max_steps = 200  # demo cap: keep smoke runs lightweight
    try:
        stats = run_episode(policy, env, max_steps=max_steps)
        print(f"\n   Episode done: steps={stats['steps']}, done={stats['done']}")
        for agent in stats["total_reward"]:
            print(f"   {agent}: reward={stats['total_reward'][agent]:.3f}  "
                  f"cost={stats['total_cost'][agent]:.3f}")
    except Exception as e:
        print(f"\n   FAIL during episode: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        env.close()

    print()
    print("=" * 60)
    print("=== END-TO-END DEMO COMPLETE ===")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

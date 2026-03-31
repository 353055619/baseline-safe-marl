"""
scripts/demo_episode.py — Minimal End-to-End Episode Demo
===============================================================
目标：验证 MAPPO / MAPPO-L / HAPPO / MACPO stub + fallback CostWrapper + safe_mamujoco_adapter
      能完整跑通 1 episode，支持算法切换。

⚠️ 这是集成验证 demo，不是训练脚本。随机策略，不做学习。

CLI 用法:
    cd /path/to/baseline-safe-marl
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MAPPO --max-steps 100
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MAPPO-L --max-steps 200
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo HAPPO --max-steps 100
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MACPO --max-steps 100
"""

import argparse
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


# --------------------------------------------------------------------
# Env factory map
# --------------------------------------------------------------------
_FALLBACK_ENV_FACTORIES = {
    "safeant2x4": make_safe_ant_2x4,
    "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
    "safehopper2": make_safe_hopper_2,
}


def resolve_policy_class(algo_name: str):
    """动态 import 对应的 Policy 类（MAPPO / MAPPO-L / HAPPO / MACPO）。"""
    name = algo_name.upper()
    if name == "MAPPO-L":
        from algos.mappo_lagrangian import MAPPOLPolicy as Policy
    elif name == "MAPPO":
        from algos.mappo import MAPPOPolicy as Policy
    elif name == "HAPPO":
        from algos.happo import HAPPOPolicy as Policy
    elif name == "MACPO":
        from algos.macpo import MACPOPolicy as Policy
    else:
        raise ValueError(f"Unsupported algo: {algo_name}. Supported: MAPPO, MAPPO-L, HAPPO, MACPO")
    return Policy


def run_episode(policy, env, max_steps: int) -> dict:
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
            print(f"  step {step_count:3d}: done={done} rewards={r_vals} costs={c_vals}")

    return {"steps": step_count, "total_reward": total_reward, "total_cost": total_cost, "done": done}


def main():
    parser = argparse.ArgumentParser(description="MAPPO/MAPPO-L end-to-end episode demo")
    parser.add_argument("--algo", type=str, default=None, choices=["MAPPO", "MAPPO-L", "HAPPO", "MACPO"],
                        help="Algorithm to run (default: from YAML config)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max episode steps (default: 200)")
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("MAPPO / MAPPO-L + Fallback Safe MAMujoco — Episode Demo")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # 1. Load YAML config
    print("\n[1] Loading config...")
    try:
        base_cfg = load_config(config_path="configs/phase1_default.yaml")
    except FileNotFoundError:
        print("  WARN: YAML not found, using hardcoded fallback cfg")
        base_cfg = {
            "algo": {"algo_name": "MAPPO", "hidden_dim": 64, "activation": "tanh", "lr": 3e-4},
            "env": {"env_name": "safehalfcheetah2x3", "fallback": True, "render_mode": None},
        }

    # Resolve effective algo: CLI override > YAML
    yaml_algo = base_cfg.get("algo", {}).get("algo_name", "MAPPO")
    effective_algo = args.algo or yaml_algo
    cfg = make_algo_config(effective_algo, base_cfg)

    env_name_raw = cfg.get("env", {}).get("env_name", "safeant2x4").lower()
    render_mode = cfg.get("env", {}).get("render_mode", None)
    algo_in_cfg = cfg.get("algo", {}).get("algo_name", "MAPPO")

    print(f"   YAML algo: {yaml_algo}")
    if args.algo:
        print(f"   CLI override: --algo={args.algo}  (effective: {effective_algo})")
    print(f"   env={env_name_raw}  max_steps={args.max_steps}")

    # 2. Create env
    print("\n[2] Creating fallback env...")
    factory = _FALLBACK_ENV_FACTORIES.get(env_name_raw) or make_safe_halfcheetah_2x3
    try:
        env = factory(render_mode=render_mode)
        print("   env created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    obs_sample, _ = env.reset(seed=42)
    sa = env.agents[0]
    obs_dim = int(np.array(obs_sample[sa]).shape[-1])
    action_dim = int(np.array(env.action_space(sa).sample()).shape[-1])
    env.close()
    cfg["env"]["obs_dim"] = obs_dim
    cfg["env"]["action_dim"] = action_dim
    print(f"   obs_dim={obs_dim}, action_dim={action_dim}")

    # 3. Create Policy (resolved from effective_algo)
    print(f"\n[3] Creating {effective_algo} policy...")
    try:
        PolicyCls = resolve_policy_class(effective_algo)
        policy = PolicyCls(cfg, agent_id=0)
        print(f"   {PolicyCls.__name__} created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback; traceback.print_exc()
        return 1

    # 4. Run episode
    print(f"\n[4] Running 1 episode (max_steps={args.max_steps})...")
    try:
        env = factory(render_mode=render_mode)  # re-create for fresh state
        stats = run_episode(policy, env, max_steps=args.max_steps)
        print(f"\n   Episode done: steps={stats['steps']}, done={stats['done']}")
        for agent in stats["total_reward"]:
            print(f"   {agent}: reward={stats['total_reward'][agent]:.3f}  "
                  f"cost={stats['total_cost'][agent]:.3f}")
        env.close()
    except Exception as e:
        print(f"\n   FAIL during episode: {e}")
        import traceback; traceback.print_exc()
        return 1

    print()
    print("=" * 60)
    print("=== END-TO-END DEMO COMPLETE ===")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

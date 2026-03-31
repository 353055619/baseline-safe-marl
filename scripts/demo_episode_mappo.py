"""
scripts/demo_episode_mappo.py — Minimal End-to-End Episode Demo
================================================================
目标：验证 MAPPO stub + fallback CostWrapper + safe_mamujoco_adapter
      能完整跑通 1 episode（reset → 多 step → episode done）。

⚠️ 这是集成验证 demo，不是训练脚本。随机策略，不做学习。

算法选择理由：
  - MAPPO：on-policy 最简，dict obs 原生支持，无 replay buffer
  - 不选 MATD3/FACMAC（off-policy 需要 buffer，stub 是 no-op）
  - 不选 MAPPO-L（多一层 lagrangian，本 demo 聚焦链路验证）

环境选择：HalfCheetah 2-agent（最简配置）

Usage:
    cd /path/to/baseline-safe-marl
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode_mappo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from envs.safe_mamujoco_adapter import make_safe_halfcheetah_2x3
from algos.mappo import MAPPOPolicy


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
ENV_CFG = {
    "device": "cpu",
    "algo": {
        "hidden_dim": 64,
        "activation": "tanh",
        "lr": 3e-4,
    },
    "env": {
        "env_name": "safehalfcheetah2x3",
        "obs_dim": None,   # auto-detect
        "action_dim": None,
    },
}


def run_episode(policy: MAPPOPolicy, env, max_steps: int = 200) -> dict:
    """Run one episode. Returns dict of collected stats."""
    obs_dict, info_dict = env.reset(seed=42)
    total_reward = {agent: 0.0 for agent in env.agents}
    total_cost = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    print(f"\nEpisode start: {len(env.agents)} agents, max_steps={max_steps}")
    print(f"  obs keys: {list(obs_dict.keys())}")

    while not done and step_count < max_steps:
        # MAPPO get_actions supports dict obs natively
        action_dict = {}
        for agent in env.agents:
            action_dict[agent] = policy.get_actions(obs_dict[agent], deterministic=True)

        obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)

        # Accumulate
        for agent in env.agents:
            total_reward[agent] += rewards.get(agent, 0.0)
            total_cost[agent] += info_dict.get("cost", {}).get(agent, 0.0)

        step_count += 1
        done = all(terms.values()) or all(truncs.values())

        if step_count % 50 == 0 or done:
            print(f"  step {step_count:3d}: agents_done={sum(terms.values())}/{len(terms)}, "
                  f"rewards={[f'{total_reward[a]:.1f}' for a in env.agents[:2]]} "
                  f"costs={[f'{total_cost[a]:.1f}' for a in env.agents[:2]]}")

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

    # 1. Create env
    print("\n[1] Creating fallback safe HalfCheetah 2x3 env...")
    try:
        env = make_safe_halfcheetah_2x3(render_mode=None)
        print("   env created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    # 2. Auto-detect obs/action dim from env
    obs_sample, _ = env.reset(seed=42)
    sample_agent = env.agents[0]
    obs_dim = int(np.array(obs_sample[sample_agent]).shape[-1])
    action_dim = int(np.array(env.action_space(sample_agent).sample()).shape[-1])
    env.close()

    ENV_CFG["env"]["obs_dim"] = obs_dim
    ENV_CFG["env"]["action_dim"] = action_dim
    print(f"   obs_dim={obs_dim}, action_dim={action_dim}")

    # 3. Create MAPPO policy
    print("\n[2] Creating MAPPO policy...")
    try:
        policy = MAPPOPolicy(ENV_CFG, agent_id=0)
        print("   MAPPOPolicy created OK")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    # 4. Run episode
    print("\n[3] Running 1 episode...")
    try:
        stats = run_episode(policy, env, max_steps=200)
        print(f"\n   Episode done: steps={stats['steps']}, done={stats['done']}")
        for agent in stats["total_reward"]:
            print(f"   {agent}: reward={stats['total_reward'][agent]:.3f}, "
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

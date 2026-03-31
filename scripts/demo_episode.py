"""
scripts/demo_episode.py — Minimal End-to-End Episode Demo
===============================================================
目标：验证 MAPPO / MAPPO-L / HAPPO / MACPO / MATD3 stub + fallback CostWrapper + safe_mamujoco_adapter
# 并支持 --train 模式：trajectory 收集 → policy.train() 演示
      能完整跑通 1 episode，支持算法切换。

⚠️ 这是集成验证 demo，不是训练脚本。随机策略，不做学习。

CLI 用法:
    cd /path/to/baseline-safe-marl
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MAPPO --max-steps 100
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MAPPO-L --max-steps 200
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo HAPPO --max-steps 100
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MACPO --max-steps 100
    uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MATD3 --max-steps 100
"""

import argparse
import csv
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


def resolve_trainer_class(algo_name: str):
    """动态 import 对应的 Trainer 类。"""
    name = algo_name.upper()
    if name == "MAPPO-L":
        from algos.mappo_lagrangian import MAPPOLTrainer as Trainer
    elif name == "MAPPO":
        from algos.mappo import MAPPOTrainer as Trainer
    elif name == "HAPPO":
        from algos.happo import HAPPOTrainer as Trainer
    elif name == "MACPO":
        from algos.macpo import MACPOTrainer as Trainer
    elif name == "MATD3":
        from algos.matd3 import MATD3Trainer as Trainer
    else:
        raise ValueError(f"Unsupported algo: {algo_name}. Supported: MAPPO, MAPPO-L, HAPPO, MACPO, MATD3")
    return Trainer


def resolve_policy_class(algo_name: str):
    """动态 import 对应的 Policy 类（MAPPO / MAPPO-L / HAPPO / MACPO / MATD3）。"""
    name = algo_name.upper()
    if name == "MAPPO-L":
        from algos.mappo_lagrangian import MAPPOLPolicy as Policy
    elif name == "MAPPO":
        from algos.mappo import MAPPOPolicy as Policy
    elif name == "HAPPO":
        from algos.happo import HAPPOPolicy as Policy
    elif name == "MACPO":
        from algos.macpo import MACPOPolicy as Policy
    elif name == "MATD3":
        from algos.matd3 import MATD3Policy as Policy
    else:
        raise ValueError(f"Unsupported algo: {algo_name}. Supported: MAPPO, MAPPO-L, HAPPO, MACPO, MATD3")
    return Policy


def collect_trajectory(policy, env, max_steps: int) -> dict:
    """
    Collect one episode of trajectory data for training demonstration.
    Returns dict with keys: obs, actions, rewards, dones, info per agent.
    """
    obs_dict, info_dict = env.reset(seed=42)
    trajectory = {agent: {"obs": [], "actions": [], "rewards": [], "dones": []} for agent in env.agents}
    step_count = 0
    done = False

    print(f"\nTrajectory collection: {len(env.agents)} agents, max_steps={max_steps}")
    while not done and step_count < max_steps:
        action_dict = {}
        for agent in env.agents:
            action_dict[agent] = policy.get_actions(obs_dict[agent], deterministic=False)
        obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)
        for agent in env.agents:
            trajectory[agent]["obs"].append(obs_dict[agent])
            trajectory[agent]["actions"].append(action_dict[agent])
            trajectory[agent]["rewards"].append(rewards.get(agent, 0.0))
            done_agent = terms.get(agent, False) or truncs.get(agent, False)
            trajectory[agent]["dones"].append(done_agent)
        step_count += 1
        done = all(terms.values()) or all(truncs.values())

    print(f"  Collected {step_count} transitions for {len(env.agents)} agents")
    return {"steps": step_count, "trajectory": trajectory}


def run_episode(policy, env, max_steps: int) -> dict:
    """Run one episode. Returns collected stats (no trajectory stored)."""
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
    parser.add_argument("--algo", type=str, default=None, choices=["MAPPO", "MAPPO-L", "HAPPO", "MACPO", "MATD3"],
                        help="Algorithm to run (default: from YAML config)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max episode steps (default: 200)")
    parser.add_argument("--train", action="store_true",
                        help="After episode: collect trajectory and call policy.train()")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("MAPPO / MAPPO-L + Fallback Safe MAMujoco — Episode Demo")
    if args.train:
        print("  [train mode: trajectory collection + trainer.train() demonstration]")
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

    # 4. CSV logger setup
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"{effective_algo}.csv"
    csv_header = ["algo", "episode", "reward", "cost", "steps", "done"]
    file_is_new = not csv_path.exists()

    print(f"\n[4] CSV log: {csv_path}  ({args.episodes} episode(s))")

    # 5. Episode loop
    print(f"\n[5] Running {args.episodes} episode(s)...")
    try:
        for ep in range(1, args.episodes + 1):
            episode_seed = 42 + ep  # different seed per episode
            env = factory(render_mode=render_mode)
            env.reset(seed=episode_seed)
            # run episode
            stats = run_episode(policy, env, max_steps=args.max_steps)
            avg_reward = sum(stats["total_reward"].values()) / max(len(stats["total_reward"]), 1)
            avg_cost = sum(stats["total_cost"].values()) / max(len(stats["total_cost"]), 1)
            print(f"  episode {ep}/{args.episodes}: reward={avg_reward:.3f}  cost={avg_cost:.3f}  steps={stats['steps']}")

            # write CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                if file_is_new:
                    writer.writeheader()
                    file_is_new = False
                writer.writerow({
                    "algo": effective_algo,
                    "episode": ep,
                    "reward": round(avg_reward, 4),
                    "cost": round(avg_cost, 4),
                    "steps": stats["steps"],
                    "done": stats["done"],
                })
            env.close()

            # --train: trajectory + train after each episode
            if args.train:
                env2 = factory(render_mode=render_mode)
                traj_stats = collect_trajectory(policy, env2, max_steps=args.max_steps)
                env2.close()
                n_steps = traj_stats["steps"]
                TrainerCls = resolve_trainer_class(effective_algo)
                trainer = TrainerCls(cfg, policy)
                train_metrics = trainer.train(num_steps=n_steps)
                print(f"  [train] episode {ep} train_metrics: { {k: float(v) if isinstance(v,(int,float)) else v for k,v in train_metrics.items()} }")

        print(f"\n   All {args.episodes} episodes logged to {csv_path}")

    except Exception as e:
        print(f"\n   FAIL during episode loop: {e}")
        import traceback; traceback.print_exc()
        return 1

    print()
    print("=" * 60)
    print(f"=== DONE: {args.episodes} episode(s) complete, results in {csv_path} ===")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

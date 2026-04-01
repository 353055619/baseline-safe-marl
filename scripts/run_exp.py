"""
scripts/run_exp.py — Unified experiment entry point
用法:
    python scripts/run_exp.py --algo MAPPO --episodes 10 --max-steps 500
    python scripts/run_exp.py --algo MATD3 --episodes 5 --max-steps 200 --env safeant2x4
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from baseline_safe_marl.config import load_config
from baseline_safe_marl.algo_config import make_algo_config
from baseline_safe_marl.envs.core.adapter import (
    make_safe_ant_2x4,
    make_safe_halfcheetah_2x3,
    make_safe_hopper_2,
)

_FALLBACK_ENV_FACTORIES = {
    "safeant2x4": make_safe_ant_2x4,
    "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
    "safehopper2": make_safe_hopper_2,
}


def resolve_policy_class(algo_name: str):
    name = algo_name.upper()
    if name == "MAPPO-L":
        from baseline_safe_marl.algos.on_policy.mappo_lagrangian import MAPPOLPolicy as Policy
    elif name == "MAPPO":
        from baseline_safe_marl.algos.on_policy.mappo import MAPPOPolicy as Policy
    elif name == "HAPPO":
        from baseline_safe_marl.algos.on_policy.happo import HAPPOPolicy as Policy
    elif name == "MACPO":
        from baseline_safe_marl.algos.on_policy.macpo import MACPOPolicy as Policy
    elif name == "MATD3":
        from baseline_safe_marl.algos.off_policy.matd3 import MATD3Policy as Policy
    elif name == "FACMAC":
        from baseline_safe_marl.algos.off_policy.facmac import FACMACPolicy as Policy
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")
    return Policy


def resolve_trainer_class(algo_name: str):
    name = algo_name.upper()
    if name == "MAPPO-L":
        from baseline_safe_marl.algos.on_policy.mappo_lagrangian import MAPPOLTrainer as Trainer
    elif name == "MAPPO":
        from baseline_safe_marl.algos.on_policy.mappo import MAPPOTrainer as Trainer
    elif name == "HAPPO":
        from baseline_safe_marl.algos.on_policy.happo import HAPPOTrainer as Trainer
    elif name == "MACPO":
        from baseline_safe_marl.algos.on_policy.macpo import MACPOTrainer as Trainer
    elif name == "MATD3":
        from baseline_safe_marl.algos.off_policy.matd3 import MATD3Trainer as Trainer
    elif name == "FACMAC":
        from baseline_safe_marl.algos.off_policy.facmac import FACMACTrainer as Trainer
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")
    return Trainer


def run_episode(policy, env, max_steps: int, deterministic: bool = True):
    obs_dict, info_dict = env.reset(seed=42)
    total_reward = {agent: 0.0 for agent in env.agents}
    total_cost = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    while not done and step_count < max_steps:
        action_dict = {
            agent: policy.get_actions(obs_dict[agent], deterministic=deterministic)
            for agent in env.agents
        }
        obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)
        for agent in env.agents:
            total_reward[agent] += rewards.get(agent, 0.0)
            total_cost[agent] += info_dict.get("cost", {}).get(agent, 0.0)
        step_count += 1
        done = all(terms.values()) or all(truncs.values())

    avg_reward = sum(total_reward.values()) / max(len(total_reward), 1)
    avg_cost = sum(total_cost.values()) / max(len(total_cost), 1)
    return {"steps": step_count, "reward": avg_reward, "cost": avg_cost, "done": done}


def main():
    parser = argparse.ArgumentParser(description="Safe MARL experiment runner")
    parser.add_argument("--algo", type=str, default="MAPPO",
                        choices=["MAPPO", "MAPPO-L", "HAPPO", "MACPO", "MATD3", "FACMAC"])
    parser.add_argument("--env", type=str, default="safeant2x4",
                        choices=["safeant2x4", "safehalfcheetah2x3", "safehopper2"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for results subdir")
    args = parser.parse_args()

    # Determine results subdirectory
    if args.exp_name:
        results_dir = Path("results") / args.exp_name
    else:
        results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{args.algo}.csv"
    csv_fields = ["algo", "episode", "reward", "cost", "steps", "done"]
    file_is_new = not csv_path.exists()

    print(f"=" * 60)
    print(f"baseline-safe-marl — {args.algo} experiment")
    print(f"  env={args.env}  episodes={args.episodes}  max_steps={args.max_steps}")
    print(f"=" * 60)

    # Load config
    try:
        base_cfg = load_config(config_path="configs/phase1_default.yaml")
    except FileNotFoundError:
        base_cfg = {
            "algo": {"algo_name": args.algo, "hidden_dim": 64},
            "env": {"env_name": args.env, "fallback": True},
        }

    effective_algo = args.algo
    cfg = make_algo_config(effective_algo, base_cfg)
    factory = _FALLBACK_ENV_FACTORIES[args.env]

    # Create env to get dims
    env = factory()
    obs_sample, _ = env.reset(seed=42)
    sa = env.agents[0]
    obs_dim = int(np.array(obs_sample[sa]).shape[-1])
    action_dim = int(np.array(env.action_space(sa).sample()).shape[-1])
    env.close()
    cfg["env"]["obs_dim"] = obs_dim
    cfg["env"]["action_dim"] = action_dim

    # Create policy and trainer
    PolicyCls = resolve_policy_class(args.algo)
    policy = PolicyCls(cfg, agent_id=0)
    TrainerCls = resolve_trainer_class(args.algo)
    trainer = TrainerCls(cfg, policy) if args.train else None

    # Run episodes
    for ep in range(1, args.episodes + 1):
        env = factory()
        env.reset(seed=42 + ep)
        stats = run_episode(policy, env, max_steps=args.max_steps)
        env.close()

        print(f"  episode {ep}/{args.episodes}: reward={stats['reward']:.3f}  "
              f"cost={stats['cost']:.3f}  steps={stats['steps']}")

        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            if file_is_new:
                w.writeheader()
                file_is_new = False
            w.writerow({
                "algo": args.algo,
                "episode": ep,
                "reward": round(stats["reward"], 4),
                "cost": round(stats["cost"], 4),
                "steps": stats["steps"],
                "done": stats["done"],
            })

    print(f"\nResults saved to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

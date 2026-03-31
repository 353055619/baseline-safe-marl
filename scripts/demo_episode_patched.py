import argparse
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config import load_config
from src.algo_config import make_algo_config
from envs.safe_mamujoco_adapter import (
    make_safe_ant_2x4,
    make_safe_halfcheetah_2x3,
    make_safe_hopper_2,
)

_FALLBACK_ENV_FACTORIES = {
    "safeant2x4": make_safe_ant_2x4,
    "safehalfcheetah2x3": make_safe_halfcheetah_2x3,
    "safehopper2": make_safe_hopper_2,
}


def resolve_policy_class(algo_name):
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
        raise ValueError(f"Unsupported algo: {algo_name}")
    return Policy


def resolve_trainer_class(algo_name):
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
        raise ValueError(f"Unsupported algo: {algo_name}")
    return Trainer


def run_episode_on_policy(policy, env, max_steps: int, seed: int = None):
    """On-policy episode runner (MAPPO, HAPPO, MACPO, MAPPO-L)."""
    obs_dict, info_dict = env.reset(seed=seed)
    total_reward = {agent: 0.0 for agent in env.agents}
    total_cost = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False
    while not done and step_count < max_steps:
        action_dict = {agent: policy.get_actions(obs_dict[agent], deterministic=True)
                       for agent in env.agents}
        obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)
        for agent in env.agents:
            total_reward[agent] += rewards.get(agent, 0.0)
            total_cost[agent] += info_dict.get("cost", {}).get(agent, 0.0)
        step_count += 1
        done = all(terms.values()) or all(truncs.values())
    avg_reward = sum(total_reward.values()) / max(len(total_reward), 1)
    avg_cost = sum(total_cost.values()) / max(len(total_cost), 1)
    return {"steps": step_count, "reward": avg_reward, "cost": avg_cost, "done": done}


def run_episode_off_policy(policy, trainer, env, max_steps: int, seed: int = None,
                            train_every: int = 100):
    """Off-policy episode runner with buffer feeding (MATD3)."""
    obs_dict, info_dict = env.reset(seed=seed)
    total_reward = {agent: 0.0 for agent in env.agents}
    total_cost = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    while not done and step_count < max_steps:
        action_dict = {agent: policy.get_actions(obs_dict[agent], deterministic=False)
                       for agent in env.agents}
        next_obs_dict, rewards, terms, truncs, info_dict = env.step(action_dict)

        # Build agent index dicts (0-indexed)
        agents = list(env.agents)
        obs_i = {i: obs_dict[a] for i, a in enumerate(agents)}
        act_i = {i: action_dict[a] for i, a in enumerate(agents)}
        rew_i = {i: float(rewards.get(a, 0.0)) for i, a in enumerate(agents)}
        next_i = {i: next_obs_dict[a] for i, a in enumerate(agents)}
        done_i = {i: float(terms.get(a, False) or truncs.get(a, False)) for i, a in enumerate(agents)}

        # Add transition to buffer
        trainer.add_transition(obs_i, act_i, rew_i, next_i, done_i)

        # Train if buffer ready
        if trainer.buffer.is_ready and step_count % train_every == 0:
            trainer.train(num_steps=train_every)

        for agent in env.agents:
            total_reward[agent] += rewards.get(agent, 0.0)
            total_cost[agent] += info_dict.get("cost", {}).get(agent, 0.0)

        obs_dict = next_obs_dict
        step_count += 1
        done = all(terms.values()) or all(truncs.values())

    avg_reward = sum(total_reward.values()) / max(len(total_reward), 1)
    avg_cost = sum(total_cost.values()) / max(len(total_cost), 1)
    return {"steps": step_count, "reward": avg_reward, "cost": avg_cost, "done": done}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        choices=["MAPPO", "MAPPO-L", "HAPPO", "MACPO", "MATD3"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg = load_config(config_path="configs/phase1_default.yaml")
    cfg = make_algo_config(args.algo, base_cfg)
    env_name_raw = cfg.get("env", {}).get("env_name", "safeant2x4").lower()
    factory = _FALLBACK_ENV_FACTORIES.get(env_name_raw) or make_safe_halfcheetah_2x3

    PolicyCls = resolve_policy_class(args.algo)
    policy = PolicyCls(cfg, agent_id=0)

    # Initialize trainer (MATD3 needs it for buffer feeding)
    is_off_policy = args.algo.upper() == "MATD3"
    if is_off_policy:
        TrainerCls = resolve_trainer_class(args.algo)
        trainer = TrainerCls(cfg, policy)
    else:
        trainer = None

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / (args.algo + ".csv")

    rows = []
    for run_idx in range(args.runs):
        for ep in range(1, args.episodes + 1):
            env = factory()
            seed = args.seed + run_idx * 1000 + ep

            if is_off_policy:
                stats = run_episode_off_policy(policy, trainer, env,
                                               max_steps=args.max_steps, seed=seed)
            else:
                stats = run_episode_on_policy(policy, env,
                                              max_steps=args.max_steps, seed=seed)

            env.close()
            r = round(stats["reward"], 4)
            c = round(stats["cost"], 4)
            rows.append({"algo": args.algo, "episode": ep,
                          "reward": r, "cost": c,
                          "steps": stats["steps"], "done": stats["done"]})
            print(f"  [{args.algo}] run={run_idx+1}/{args.runs} ep={ep}/{args.episodes} "
                  f"reward={r:.3f} cost={c:.3f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "episode", "reward", "cost", "steps", "done"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

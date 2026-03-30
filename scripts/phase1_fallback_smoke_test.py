"""
Phase 1 Fallback Smoke Test
============================
⚠️ 这是 fallback 验证路线，不是最终 benchmark。
结果仅用于验证集成链路可通，不作为安全性能对比依据。

功能：
    - 用 fallback adapter（gymnasium 单体 env + 逻辑多智能体 wrapper）
    - 跑 1 episode random policy，收集 episode_return / episode_cost / episode_length
    - 输出 JSON 结果到 results/phase1/fallback/

用法：
    python3.10 scripts/phase1_fallback_smoke_test.py [--env ant|halfcheetah|hopper|walker]
    推荐（显式设置 LD_LIBRARY_PATH）：
    LD_LIBRARY_PATH=/home/godw/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \\
        python3.10 scripts/phase1_fallback_smoke_test.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

# 项目根目录
# 脚本位于 /.../baseline-safe-marl/scripts/phase1_fallback_smoke_test.py
# PROJECT_ROOT = /.../baseline-safe-marl/
_SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _SCRIPT_DIR.parent   # scripts/ 的 parent = 项目根
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1" / "fallback"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 添加项目根到 sys.path，这样 "from envs.safe_mamujoco_adapter import ..." 可以工作
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_env_factory(env_name: str):
    """按名字返回对应的 fallback env factory。"""
    from envs.safe_mamujoco_adapter import (
        make_safe_ant_2x4,
        make_safe_halfcheetah_2x3,
        make_safe_hopper_2,
        make_safe_walker_2,
    )

    factories = {
        "ant": make_safe_ant_2x4,
        "halfcheetah": make_safe_halfcheetah_2x3,
        "hopper": make_safe_hopper_2,
        "walker": make_safe_walker_2,
    }
    if env_name not in factories:
        raise ValueError(f"Unknown env: {env_name}. Options: {list(factories.keys())}")
    return factories[env_name]


def run_episode(env_factory, seed: int = 42, max_steps: int = 1000):
    """
    用 random policy 跑 1 个 episode，返回汇总结果。

    Returns
    -------
    dict with keys:
        episode_return (float)
        episode_cost (float)
        episode_length (int)
        cost_violation_rate (float)
        agents (list of str)
        env_name (str)
    """
    env = env_factory()

    obs_dict, info_dict = env.reset(seed=seed)

    episode_return = 0.0
    episode_cost = 0.0
    episode_length = 0
    step_costs = []

    agents = env.agents

    for step_i in range(max_steps):
        # random policy
        actions = {
            a: env.action_space(a).sample() for a in agents
        }
        obs_dict, rewards, terms, truncs, info_dict = env.step(actions)

        # 收集 reward 和 cost（取各 agent 平均）
        for a in agents:
            episode_return += float(rewards.get(a, 0.0))
            cost_a = info_dict.get("cost", {}).get(a, 0.0)
            episode_cost += float(cost_a)
            step_costs.append(cost_a)

        episode_length += 1

        # episode 结束判断
        if any(terms.values()) or any(truncs.values()):
            break

    env.close()

    # 汇总
    n_agents = len(agents)
    avg_cost_per_step = episode_cost / max(episode_length, 1)
    cost_violation_rate = sum(1 for c in step_costs if c > 0) / max(len(step_costs), 1)

    return {
        "episode_return": round(episode_return / max(n_agents, 1), 4),
        "episode_cost": round(episode_cost / max(n_agents, 1), 4),
        "episode_length": episode_length,
        "cost_violation_rate": round(cost_violation_rate, 4),
        "total_return": round(episode_return, 4),
        "total_cost": round(episode_cost, 4),
        "agents": agents,
        "env_name": getattr(env_factory, "_env_name", "unknown"),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase1 Fallback Smoke Test")
    parser.add_argument(
        "--env",
        default="ant",
        choices=["ant", "halfcheetah", "hopper", "walker"],
        help="Fallback environment to test (default: ant)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)",
    )
    args = parser.parse_args()

    # 确定环境名（用于结果文件）
    env_display_name = f"Safe{args.env.capitalize()}-{args.env != 'halfcheetah' and '2x4' or '2x3'}"

    # 构造结果文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"smoke_{args.env}_{timestamp}"
    result_file = RESULTS_DIR / f"{run_id}_run_log.json"

    print("=" * 60)
    print("Phase 1 Fallback Smoke Test")
    print(f"⚠️  fallback prototype — not benchmark")
    print("=" * 60)
    print(f"Environment : {args.env}")
    print(f"Seed        : {args.seed}")
    print(f"Max steps   : {args.max_steps}")
    print(f"Output      : {result_file}")
    print("-" * 60)

    # 运行
    env_factory = get_env_factory(args.env)
    result = run_episode(env_factory, seed=args.seed, max_steps=args.max_steps)

    # 构造完整结果 JSON
    output = {
        **result,
        "run_id": run_id,
        "timestamp": timestamp,
        "prototype_type": "fallback",
        "is_benchmark": False,
        "warning": (
            "This is a fallback prototype run, NOT a benchmark result. "
            "Do not use for performance comparison or research claims."
        ),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "file_version": "1.0",
    }

    # 打印摘要
    print(f"Episode length : {output['episode_length']} steps")
    print(f"Episode return : {output['episode_return']}")
    print(f"Episode cost   : {output['episode_cost']}")
    print(f"Cost viol rate : {output['cost_violation_rate']}")
    print("-" * 60)
    print(f"✅ run_id: {run_id}")

    # 写入结果文件
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Results written to: {result_file}")

    # 追加到 index.json（方便汇总查看）
    index_file = RESULTS_DIR / "index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
    else:
        index = {"runs": [], "prototype_type": "fallback", "is_benchmark": False}

    index["runs"].append({
        "run_id": run_id,
        "env": args.env,
        "episode_return": output["episode_return"],
        "episode_cost": output["episode_cost"],
        "episode_length": output["episode_length"],
        "timestamp": timestamp,
        "result_file": str(result_file.name),
    })

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"✅ Index updated: {index_file}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()

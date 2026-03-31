"""
scripts/smoke_test_algos.py — Unified Algorithm Smoke Test
============================================================
最小 import + instantiate 测试，覆盖当前所有已完成的算法 stub。

Usage:
    cd /path/to/baseline-safe-marl
    uv run --with torch --with gymnasium python scripts/smoke_test_algos.py

Algorithms tested:
    - MAPPO         (algos/mappo)
    - MAPPO-L      (algos/mappo_lagrangian)
    - HAPPO        (algos/happo)
    - MACPO        (algos/macpo)
    - MATD3        (algos/matd3)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


SMOKE_TESTS = [
    {
        "name": "MAPPO",
        "module": "algos.mappo",
        "policy": "MAPPOPolicy",
        "trainer": "MAPPOTrainer",
    },
    {
        "name": "MAPPO-L (Lagrangian)",
        "module": "algos.mappo_lagrangian",
        "policy": "MAPPOLPolicy",
        "trainer": "MAPPOLTrainer",
    },
    {
        "name": "HAPPO",
        "module": "algos.happo",
        "policy": "HAPPOPolicy",
        "trainer": "HAPPOTrainer",
    },
    {
        "name": "MACPO",
        "module": "algos.macpo",
        "policy": "MACPOPolicy",
        "trainer": "MACPOTrainer",
    },
    {
        "name": "MATD3",
        "module": "algos.matd3",
        "policy": "MATD3Policy",
        "trainer": "MATD3Trainer",
    },
]

MINIMAL_CFG = {
    "device": "cpu",
    "algo": {
        "hidden_dim": 32,
        "activation": "tanh",
        "lr": 1e-3,
    },
    "env": {
        "env_name": "ant",
        "obs_dim": 27,
        "action_dim": 6,
    },
}


def test_algo(spec: dict) -> tuple[bool, str]:
    """Test a single algorithm stub. Returns (passed, message)."""
    name = spec["name"]
    try:
        mod = __import__(spec["module"], fromlist=[spec["policy"], spec["trainer"]])
        PolicyCls = getattr(mod, spec["policy"])
        TrainerCls = getattr(mod, spec["trainer"])

        policy = PolicyCls(MINIMAL_CFG, agent_id=0)
        trainer = TrainerCls(MINIMAL_CFG, policy)

        # Some stubs (MAPPOLPolicy, MACPOPolicy) don't set obs_dim in __init__;
        # use cfg fallback for smoke test.
        obs_dim = getattr(policy, "obs_dim", MINIMAL_CFG["env"]["obs_dim"])
        action_dim = getattr(policy, "action_dim", MINIMAL_CFG["env"]["action_dim"])
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = policy.get_actions(obs, deterministic=True)
        # Scalar actions (discrete stub style) are OK; just check it's a numpy array
        assert hasattr(action, "shape"), f"{name}: action has no shape attr: {action}"
        # Most stubs produce continuous actions; MAPPO-L/MACPO are scalar discrete stubs
        if action_dim > 1:
            assert action.shape == (action_dim,), \
                f"{name}: action shape {action.shape} vs {(action_dim,)}"

        result = policy.evaluate_actions(obs, action)
        assert any(k in result for k in ("log_prob", "q1", "q_min")), \
            f"{name}: evaluate_actions missing expected key"

        metrics = trainer.train(num_steps=5)
        assert isinstance(metrics, dict), f"{name}: train() did not return dict"

        trainer.update_lagrangian()

        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 60)
    print("baseline-safe-marl — Unified Algorithm Smoke Test")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    all_passed = True
    results = []

    for spec in SMOKE_TESTS:
        passed, msg = test_algo(spec)
        results.append((spec["name"], passed, msg))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {spec['name']:<30} {msg}")
        if not passed:
            all_passed = False

    print()
    print("=" * 60)
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"Summary: {passed_count}/{len(results)} algorithms passed")
    if all_passed:
        print("=== ALL SMOKE TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")
        for n, p, m in results:
            if not p:
                print(f"  FAIL: {n} — {m}")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

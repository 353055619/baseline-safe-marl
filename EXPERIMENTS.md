# Experiments

## 2026-03-31 — Phase 1 Initial Benchmark

### Scripts

**`scripts/demo_episode_patched.py`** — Main experiment runner

```bash
# Usage
python scripts/demo_episode_patched.py --algo <ALGO> --runs <N> --episodes <M> --max-steps <S>

# Examples
python scripts/demo_episode_patched.py --algo HAPPO --runs 4 --episodes 5 --max-steps 500
python scripts/demo_episode_patched.py --algo MACPO --runs 4 --episodes 5 --max-steps 500
python scripts/demo_episode_patched.py --algo MATD3 --runs 4 --episodes 5 --max-steps 500
```

Outputs CSV to `results/{ALGO}.csv` with columns: `algo, episode, reward, cost, steps, done`

**Environment**: `conda activate safe-marl` (Python 3.10, mujoco 3.6.0)

### Algorithms

| Algo | Policy Type | Trainer | Buffer |
|------|-------------|---------|--------|
| MAPPO | On-policy stochastic | MAPPOTrainer | None (on-policy) |
| MAPPO-L | On-policy stochastic + Lagrangian | MAPPOLTrainer | None |
| HAPPO | On-policy stochastic (ACKTR trust region) | HAPPOTrainer | None |
| MACPO | On-policy stochastic + Lagrangian | MACPOTrainer | None |
| MATD3 | Off-policy deterministic + twin critics | MATD3Trainer | MultiAgentReplayBuffer |

### Results (SafeAnt2x4, max_steps=500, random policy)

| Algo | Mean Reward | Mean Cost | Runs × Eps |
|------|------------|-----------|------------|
| MAPPO-L | ~192 | 200 | 4×5 |
| MACPO | ~497 | 500 | 2×5 |
| HAPPO | ~477 | 500 | 2×5 |
| MAPPO | ~457 | 200 | 4×5 |
| MATD3 | ~196 | 200 | 1×2 (early) |

Note: MATD3 results are early-stage (insufficient training steps, buffer barely full).

### Server Info

- **Server 2** (`ubuntu-server-2`): `godw@172.20.135.15 -p 20022`
  - baseline-safe-marl at `~/code/baseline-safe-marl`
  - Results at `~/code/baseline-safe-marl/results/`

### Key Implementation Notes

1. **MATD3 ReplayBuffer** (`algos/matd3/replay_buffer.py`): Multi-agent buffer storing per-agent transitions. Sample returns dict of tensors keyed by agent index.

2. **Off-policy vs On-policy**: MATD3 uses `run_episode_off_policy()` which feeds transitions to buffer and calls `trainer.train()` every `train_every=100` steps. All others use `run_episode_on_policy()`.

3. **CSV output**: All algos produce `results/{ALGO}.csv` — these are the canonical results for plotting.

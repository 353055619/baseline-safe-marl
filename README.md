# baseline-safe-marl

Multi-agent Safe RL baseline project — 6 algorithms on multi-agent safe MuJoCo.

## Current Status

### Phase 1 — Prototype (active)

**Fallback prototype: ✅ complete**
- `envs/fallback_cost_wrapper.py` — cost signal wrapper for gymnasium envs
- `envs/safe_mamujoco_adapter.py` — PettingZoo-style multi-agent adapter (gymnasium backend)
- `scripts/phase1_fallback_smoke_test.py` — 1-episode integration smoke test
- `results/phase1/fallback/` — smoke test run logs (4 envs, random policy)

⚠️ **This is a fallback prototype for adapter verification, NOT the final benchmark.**

**Native Safe MAMujoco: ⏳ blocked**
- `docs/environment-smoke-test-plan.md`
- Root cause: `mujoco_py + MuJoCo 2.1.0 binary` not installable on server-2 (Python 3.14 agv env)
- Status: investigating alternative installation paths; fallback bridge used for Phase 1

### Phase 1 文档

| 文档 | 内容 |
|------|------|
| `docs/architecture/phase1-prototype-plan.md` | Phase 1 原型计划（原生路线） |
| `docs/architecture/phase1-fallback-plan.md` | Phase 1 fallback 计划（当前路线） |
| `docs/environment-smoke-test-plan.md` | 环境 smoke test 记录与阻塞分析 |
| `docs/algorithm-research.md` | 6 个算法候选调研与 shortlist |

## Principles

- Team collaboration first
- Spec-Driven Development
- Start from small demos, then iterate
- Prefer reuse over reinventing the wheel
- Keep code, docs, experiments, and paper organized

## Project Structure

```
baseline-safe-marl/
├── configs/              # YAML experiment configs
├── docs/
│   ├── architecture/     # Phase plans, specs
│   ├── specs/            # Project foundation spec
│   └── algorithm-research.md
├── envs/
│   ├── fallback_cost_wrapper.py    # cost signal wrapper
│   └── safe_mamujoco_adapter.py   # multi-agent adapter (PettingZoo API)
├── results/phase1/fallback/        # smoke test outputs
├── scripts/
│   └── phase1_fallback_smoke_test.py
├── src/                  # shared utilities (config loader, logger, etc.)
└── third_party/          # external references (read-only)
```

## Quick Start (Server-2)

```bash
# Server environment (Python 3.10 required for mujoco support)
ssh godw@172.20.135.15 -p 20022
LD_LIBRARY_PATH=/home/godw/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    /usr/bin/python3.10 /home/godw/code/phd2/scripts/phase1_fallback_smoke_test.py --env ant
```

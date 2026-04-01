# Smoke Test Quickstart

## 用途

统一 smoke test 验证 baseline-safe-marl 的核心组件是否可运行：
- `fallback_cost_wrapper.py` — cost signal wrapper
- `safe_mamujoco_adapter.py` — multi-agent env adapter
- `phase1_fallback_smoke_test.py` — 1-episode integration test

**这是最小可运行性验证，不是完整训练或 benchmark。**

---

## 入口脚本

```
scripts/phase1_fallback_smoke_test.py
```

---

## 最小运行命令（服务器）

```bash
# 服务器 Python 3.10 + mujoco
ssh godw@172.20.135.15 -p 20022
LD_LIBRARY_PATH=/home/godw/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    /usr/bin/python3.10 /home/godw/code/phd2/scripts/phase1_fallback_smoke_test.py --env ant
```

可选 `--env`：`ant` | `halfcheetah` | `hopper` | `walker`（默认 ant）

---

## 输出说明

运行后显示 `[PASS]` / `[FAIL]`，末尾打印：

| 字段 | 含义 |
|------|------|
| `episode_return` | 该 episode 的累计 reward（随机策略，值本身无意义） |
| `episode_cost` | 该 episode 的累计 cost |
| `episode_length` | 步数 |
| `cost_violation_rate` | cost > 0 的步数占比 |
| `is_benchmark` | 恒为 `false` |

结果写入 `results/phase1/fallback/smoke_<env>_<timestamp>_run_log.json`，JSON 头含 `warning: "fallback prototype — not benchmark"`。

---

## 注意事项

- 运行失败 → 检查 `LD_LIBRARY_PATH` 是否包含 mujoco 二进制路径
- 本地 Mac（无 gymnasium）无法运行，需在服务器上执行
- smoke test 只验证 env + adapter + random policy 链路，不涉及任何学习

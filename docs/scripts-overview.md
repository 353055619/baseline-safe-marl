# Scripts Overview

> 当前项目中所有可运行脚本的快速索引。面向维护者。
> 最后更新：2026-03-31

---

## 脚本清单

| 脚本 | 用途 | 定位 |
|------|------|------|
| `scripts/smoke_test_algos.py` | 验证 6 个 stub 可 import + instantiate | Stub 验收 |
| `scripts/phase1_fallback_smoke_test.py` | 验证 fallback env adapter 可跑 1 episode | Env adapter 验收 |
| `scripts/demo_episode.py` | 验证 stub + env + trainer 完整链路，可切换算法 | 集成 demo |

---

## 入口命令

### 1. `smoke_test_algos.py`（本地/Mac 可跑）

```bash
cd ~/Documents/godclaw/github/baseline-safe-marl
uv run --with torch --with gymnasium python scripts/smoke_test_algos.py
```

**预期输出关键词**：`MAPPO ... ✅` `MATD3 ... ✅` `FACMAC ... ✅` 等，末尾 `6/6 PASS`

---

### 2. `phase1_fallback_smoke_test.py`（服务器，需 python3.10 + mujoco）

```bash
ssh godw@172.20.135.15 -p 20022
LD_LIBRARY_PATH=/home/godw/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    /usr/bin/python3.10 /home/godw/code/phd2/scripts/phase1_fallback_smoke_test.py --env ant
```

可选 `--env`：ant | halfcheetah | hopper | walker（默认 ant）

**预期输出关键词**：`[PASS] Ant-2x4` → `episode_return:` `episode_cost:` `episode_length:`

---

### 3. `demo_episode.py`（本地/Mac 可跑）

```bash
cd ~/Documents/godclaw/github/baseline-safe-marl
uv run --with torch --with gymnasium --with numpy python scripts/demo_episode.py --algo MAPPO-L --max-steps 200
```

可选 `--algo`：MAPPO | MAPPO-L | HAPPO | MACPO | MATD3 | FACMAC
可选 `--max-steps`：单 episode 最大步数（默认 200）

**预期输出关键词**：`episode_return:` `policy_loss:` `entropy:` 或 `q_tot:`

---

## 脚本关系

```
smoke_test_algos.py          # 最浅：stub 能否 import
    ↓
phase1_fallback_smoke_test.py  # 中：env adapter 是否可跑
    ↓
demo_episode.py             # 最深：stub + env + trainer 完整链路
    ↓
（未来）最小训练脚本          # 加入 rollout buffer + 多 episode 循环
```

从左到右：smoke test → demo → 最小训练，验证深度递增、工程复杂度递增。

---

## 注意事项

- `phase1_fallback_smoke_test.py` 需要服务器（python3.10 + mujoco），本地 Mac 无法运行
- `smoke_test_algos.py` 和 `demo_episode.py` 可在本地通过 `uv run` 执行
- 所有脚本均为**验证性**，不执行真实学习

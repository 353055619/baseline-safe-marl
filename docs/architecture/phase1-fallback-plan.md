# Phase 1 Fallback Plan

> 文档版本：v1.0
> 作者：phd2
> 日期：2026-03-30
> 状态：Fallback 备选方案，仅在原生 Safe MAMujoco 无法在服务器上运行时触发
> 触发条件：Safe MAMujoco 安装阻塞超过预期时间，或环境适配在 2 小时内无法通过 smoke test

---

## 1. 背景与前提

**已知事实（截至 2026-03-30）：**
- 服务器（RTX 3090, Python 3.10）：`gymnasium + mujoco` 可跑通单体 MuJoCo smoke test ✅
- Safe MAMujoco（chauncygu/Safe-Multi-Agent-Mujoco）依赖 `mujoco_py + MuJoCo 2.1.0 binary`，服务器上卡住 ❌

**本文档仅回答：**
> 如果原生 Safe MAMujoco 暂时跑不通，phase1 如何用 gymnasium 原生环境 + 自定义 cost wrapper + multi-agent adapter 跑出一个"可演示 prototype"，同时明确这不是最终 benchmark。

---

## 2. Fallback Prototype 的目标与边界

### 目标
- 在服务器上验证：multi-agent adapter → MAPPO/MAPPO-L stub → cost-aware rollout → 结果记录，这条集成链路是可跑的
- 给 Godw 一个**可演示的视觉输出**（哪怕是简化版的）

### 边界（红线）
| 可以 | 不可以 |
|------|--------|
| 演示 cost-aware rollout 链路 | 声称这是 Safe MAMujoco benchmark |
| 与原生 benchmark 共享接口层 | 声称结果是真实安全性能对比 |
| 用简化环境做 adapter 验证 | 跑完整训练（< 1h 是上限） |
| 记录结果到 `results/phase1/fallback/` | 写 paper-level 实验报告 |

### 与原生 benchmark 的接口对齐要求
即便用 fallback，adapter 接口必须与原生 Safe MAMujoco 保持一致，保证未来切换时算法代码不需要改：

```python
# 必须是这个签名，未来换原生环境时算法代码无感知
def make_safe_ant_2x4(render_mode="rgb_array"):
    """返回 gymnasiumParallelEnv，接口与原生 Safe MAMujoco 对齐"""
    ...

class SafeMAMujocoAdapter:
    def reset(self, seed=None) -> (obs, info)
    def step(self, actions: dict) -> (obs, reward, terminated, truncated, info)
    # info 必须含 cost 字段，与原生一致
    # obs 格式与原生一致（保证 policy 不需要改）
```

---

## 3. Fallback 环境构造方案

### 构造方式

用 gymnasium 原生 multi-agent 环境 + 自定义 cost wrapper：

```python
# envs/fallback_safe_adapter.py

class CostWrapper(gymnasium.Wrapper):
    """
    在任意 gymnasium 多智能体环境上叠加 cost 信号。
    逻辑参考 Safe MAMujoco：越接近危险区域 cost=1，否则 cost=0。
    """
    def __init__(self, env, cost_fn=None):
        super().__init__(env)
        self.cost_fn = cost_fn  # 可自定义 cost 规则

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["cost"] = 0.0
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # 自定义 cost 计算逻辑
        info["cost"] = self.cost_fn(self.env, obs, actions) if self.cost_fn else 0.0
        return obs, reward, terminated, truncated, info
```

### Fallback 环境清单（优先级从高到低）

| 环境 | 构造方式 | 目的 |
|------|---------|------|
| `HalfCheetah-v5` + `CostWrapper` | gymnasium 原生 | 连续控制 smoke test，最快验证 |
| `Ant-v5` + `CostWrapper` | gymnasium 原生 | 与最终 benchmark 任务最接近 |
| `multiagent_mujoco` (PettingZoo) + CostWrapper | third_party | 多智能体场景 adapter 验证 |

---

## 4. 最少 3 个小任务

### Fallback Task 1：CostWrapper 实现
**负责人：phd1**
**输入：** gymnasium ParallelEnv API 规范 + Safe MAMujoco cost 信号定义参考
**输出：** `envs/fallback_cost_wrapper.py`
**内容：**
- `CostWrapper` 类：接受任意 gymnasium ParallelEnv，添加 cost 字段
- 提供预设 cost 函数（基于 position/velocity 阈值判断）
- smoke test：HalfCheetah + CostWrapper → reset/step 100 步不报错

**验收标准：**
```
python -c "
from envs.fallback_cost_wrapper import CostWrapper, make_halfcheetah_cost
import gymnasium as gym
env = CostWrapper(gym.make('HalfCheetah-v5'), cost_fn=make_halfcheetah_cost())
obs, info = env.reset(seed=42)
for i in range(100):
    actions = env.action_space.sample()
    obs, r, term, trunc, info = env.step(actions)
    assert 'cost' in info
env.close()
print('CostWrapper OK')
"
```

---

### Fallback Task 2：Multi-Agent Adapter Stub（与原生接口对齐）
**负责人：phd2**
**输入：** 原生 Safe MAMujoco 的接口定义（reset/step/info 格式）+ CostWrapper
**输出：** `envs/safe_mamujoco_adapter.py`（fallback 版，先不依赖原生库）
**内容：**
- `make_safe_ant_2x4()`：返回 ParallelEnv + CostWrapper，接口签名与原生版本一致
- `make_safe_halfcheetah_2x3()` 同理
- `SafeMAMujocoAdapter` 类：统一接口，future-proof

**验收标准：**
```
python -c "
from envs.safe_mamujoco_adapter import make_safe_ant_2x4, make_safe_halfcheetah_2x3

env = make_safe_ant_2x4()
obs, info = env.reset(seed=42)
for i in range(50):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
    assert 'cost' in info
env.close()
print('fallback adapter OK')
"
```

---

### Fallback Task 3：Phase1 Fallback Smoke Test
**负责人：phd2**
**输入：** Task 1 + Task 2 产出物 + configs/phase1_default.yaml
**输出：** `scripts/phase1_fallback_smoke_test.py` + `results/phase1/fallback/`
**内容：**
- 随机策略在 `make_safe_ant_2x4()` 上跑 1 episode
- 记录：episode_return, episode_cost, episode_length, cost_violation_rate
- **标注**：在脚本 docstring 和输出文件头明确写明这是 fallback 验证路线

**输出文件头（必须包含）：**
```python
"""
Phase 1 Fallback Smoke Test
⚠️ 警告：这是 fallback 验证路线，不是最终 benchmark。
结果仅用于验证集成链路可通，不作为安全性能对比依据。
"""
```

**验收标准：**
```
xvfb-run -a python scripts/phase1_fallback_smoke_test.py
# 期望：脚本正常退出
# results/phase1/fallback/smoke_YYYYMMDD/run_log.json 包含：
#   - episode_return
#   - episode_cost
#   - episode_length
#   - cost_violation_rate
#   - warning: "fallback prototype - not benchmark"
```

---

## 5. 文档标注规范（如何声明这是 fallback）

在以下位置必须明确标注：

| 位置 | 标注方式 |
|------|---------|
| `results/phase1/fallback/*/run_log.json` | JSON 头加 `"prototype_type": "fallback", "is_benchmark": false` |
| `scripts/phase1_fallback_smoke_test.py` | 文件头 docstring 加 ⚠️ 警告 |
| `docs/architecture/phase1-fallback-plan.md` | 本文档标题下加状态标签 |
| `README.md` | 在 Phase 1 进度部分加一行说明，链接到 fallback-plan |

示例 README 片段：
```markdown
## Phase 1 Status

- [x] Env smoke test (gymnasium + mujoco) ✅
- [ ] Native Safe MAMujoco adapter ⏳ (blocked by mujoco_py)
- [x] Fallback prototype ⚠️ [docs/architecture/phase1-fallback-plan.md]

> ⚠️ Fallback prototype 是 adapter 验证路线，不是最终 benchmark。结果不作为安全性能对比依据。
```

---

## 6. Fallback 局限性声明

| 局限性 | 影响 |
|--------|------|
| gymnasium 原生环境不是真正的 multi-agent 机器人控制 | 智能体间无物理耦合，奖励函数完全不同 |
| CostWrapper 是人为规则，非物理仿真 | cost 信号真实性远低于 Safe MAMujoco |
| 单 episode 演示，无法评估学习效果 | 不做训练，无法验证算法收敛性 |
| 不代表 Safe MAMujoco benchmark 结果 | 任何基于此原型的外推都是无效的 |

---

## 7. 任务依赖

```
Fallback Task 1（CostWrapper）  ←最先，无依赖
         ↓
Fallback Task 2（Adapter Stub） ←依赖 Task 1
         ↓
Fallback Task 3（Smoke Test）  ←依赖 Task 1+2
```

---

*本文档为 fallback 专用，待原生 Safe MAMujoco 可跑通后本文档失效，以 phase1-prototype-plan.md 为准。*

---

## 实操记录（2026-03-30）

### 服务器环境差异
- agv 环境（Python 3.14）：mujoco Python 绑定无预编译 wheel，`pip install mujoco` 和 `gymnasium[mujoco]` 均失败
- 解决：使用系统 Python 3.10 + `gymnasium[mujoco]`（conda-forge 补充 mujoco 二进制）
- 推荐执行命令：
  ```bash
  LD_LIBRARY_PATH=/home/godw/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
      /usr/bin/python3.10 /path/to/phase1_fallback_smoke_test.py --env ant
  ```

### envs/ 目录要求
- `envs/` 必须有 `__init__.py` 才能被 Python 作为 package 导入
- 已在本地和服务器同步创建：`envs/__init__.py`

### 项目路径结构（服务器）
```
/home/godw/code/phd2/
├── scripts/
│   └── phase1_fallback_smoke_test.py   ← 执行入口
├── envs/
│   ├── __init__.py                     ← 必须存在
│   ├── fallback_cost_wrapper.py
│   └── safe_mamujoco_adapter.py
└── results/
    └── phase1/
        └── fallback/                   ← smoke test 结果
```

### 实际 smoke test 结果（2026-03-30）

| 环境 | episode_return | episode_cost | episode_length | cost_viol_rate |
|------|----------------|--------------|----------------|----------------|
| Ant-2x4 | 62.42 | 155.0 | 163 | 0.951 |
| HalfCheetah-2x3 | -183.27 | 0.0 | 1000 | 0.0 |
| Hopper-2 | 25.44 | 0.0 | 35 | 0.0 |
| Walker-2 | 7.56 | 0.0 | 21 | 0.0 |

⚠️ 以上为 random policy fallback 原型结果，不是 benchmark。

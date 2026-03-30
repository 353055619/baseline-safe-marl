# Phase 1 Prototype Plan

> 文档版本：v1.0
> 作者：phd2
> 日期：2026-03-30
> 状态：Draft，待 Godclaw review
> 依据：docs/algorithm-research.md + docs/specs/project-foundation.md

---

## 1. 为什么第一阶段先做 MAPPO-L + MAPPO（不是 6 个）

**核心原因：这两个算法是同一代码库，共享 80%+ 实现，天然构成配对。**

| 对比维度 | MAPPO | MAPPO-L |
|---------|-------|---------|
| 安全性 | ❌ 非安全基线 | ✅ Lagrangian 约束层 |
| 训练代码 | 100% 共享 | +约束损失 +拉格朗日乘子更新 |
| 已有 benchmark 证明 | ✅ Safe MAMujoco 官方验证 | ✅ 同上 |
| 工程接入成本 | 低 | 低（略高于 MAPPO） |

同时跑这一对的理由：
1. **一次 smoke test 验证两个算法** — 共享环境 + 训练框架，适配成本只花一次
2. **安全 vs 非安全对比在原型阶段就有意义** — 比后期加基线更干净
3. **其余 4 个（MACPO/HAPPO/MATD3/FACMAC）依赖不同代码路径或 actor 架构** — 放在 phase 2 减少当前工程复杂度
4. **符合 spec 要求（SC-001/008）** — 先跑通一个可演示的原型路径，不追求全量 baseline

---

## 2. Phase 1 最小 Repo 架构

```
baseline-safe-marl/
├── configs/                  # YAML 配置（env / algo / hyperparams）
│   └── phase1_default.yaml
├── docs/
│   ├── architecture/
│   │   └── phase1-prototype-plan.md   ← 本文档
│   └── environment-smoke-test-plan.md
├── envs/                     # Benchmark adapter（与算法解耦）
│   ├── __init__.py
│   └── safe_mamujoco_adapter.py   # Gymnasium wrapper for Safe MAMujoco
├── algos/                    # 算法实现 stubs
│   ├── __init__.py
│   ├── base.py               # MARL algo 基类（抽象接口）
│   ├── mappo/                # MAPPO（非安全基线）
│   │   ├── __init__.py
│   │   ├── trainer.py        # trainer stub
│   │   └── policy.py         # policy stub
│   └── mappo_lagrangian/     # MAPPO-L（安全版本）
│       ├── __init__.py
│       ├── trainer.py
│       └── policy.py
├── scripts/                  # 可执行脚本
│   ├── phase1_smoke_test.py  # 最小集成测试（1 episode，随机策略）
│   └── train_mappo_l.py     # 实际训练脚本（stub，不跑全量）
├── results/                  # 实验结果
│   └── phase1/               # phase1 smoke test 输出
│       └── smoke_run_YYYYMMDD/
├── smoke_test/               # 独立 smoke test 脚本
│   └── test_env_adapter.py
├── src/                      # 共享工具（logger / config loader / replay buffer 等）
│   ├── __init__.py
│   └── utils.py
├── tests/                    # 单元测试
│   └── test_adapter.py
└── third_party/              # 外部引用（git submodule 或 patch）
    └── on-policy/            # marlbenchmark/on-policy 的本地引用
```

**设计原则：**
- envs/ 和 algos/ 通过抽象接口解耦，支持后续替换算法而不换 adapter
- algos/ 只放 phase1 涉及的两种算法，phase2 扩展时新增目录
- third_party/ on-policy 是只读引用，不直接修改，用于代码参考

---

## 3. 小任务清单

### Task A：Benchmark Adapter — Safe MAMujoco Gymnasium Wrapper
**负责人建议：phd1**
**输入：** Safe MAMujoco 环境（chauncygu/Safe-Multi-Agent-Mujoco）+ gymnasium API 规范
**输出：** `envs/safe_mamujoco_adapter.py` + `smoke_test/test_adapter.py`
**内容：**
- 将 Safe MAMujoco（Ant-2x4, HalfCheetah-2x3 等）包装为 gymnasium.parallel.ParallelEnv 或 gymnasium.make() 可调用的形式
- 支持 `reset(seed=)`、`step(actions)`、`close()`
- 返回格式：`obs, reward, terminated, truncated, info`（含 cost 字段）

**验收标准：**
```
# 任意一行运行不报错即通过
python -c "
from envs.safe_mamujoco_adapter import make_safe_ant_2x4
env = make_safe_ant_2x4()
obs, info = env.reset(seed=42)
for i in range(50):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
env.close()
print('smoke: env adapter OK')
"
```

---

### Task B：项目配置系统
**负责人建议：phd2**
**输入：** 项目已有目录结构 + 已有 algorithm-research.md 中的算法参数
**输出：** `configs/phase1_default.yaml` + `src/config.py`（配置加载器）
**内容：**
- YAML 中定义：env_name, algo_name, num_envs, num_steps, lr, gamma, lam, clip_eps, hidden_dim, target_kl, cost_limit 等
- config.py 提供 `load_config(path) → dict`，支持 CLI override（如 `python script.py --config.key=value`）

**验收标准：**
```
python -c "
from src.config import load_config
cfg = load_config('configs/phase1_default.yaml')
assert 'env' in cfg and 'algo' in cfg
assert cfg['env']['env_name'] == 'SafeAnt2x4'
print('config system OK:', cfg['algo']['algo_name'])
"
```

---

### Task C：MAPPO 算法 Stub
**负责人建议：phd1**
**输入：** `marlbenchmark/on-policy` 代码参考 + `configs/phase1_default.yaml`
**输出：** `algos/mappo/` 目录（含 `policy.py` 和 `trainer.py` stub）
**内容：**
- `PolicyMAPPOPartial` 类：`update(rollout_buffer)` 和 `get_actions(obs)` 接口
- `MAPPOtrainer` 类：接受 config，调用 policy，支持 `train()` 和 `save()/load()`
- **现在只实现类结构和最重要的方法签名，不跑真实训练**

**验收标准：**
```
python -c "
from algos.mappo import MAPPOPolicy, MAPPOTrainer
from src.config import load_config
cfg = load_config('configs/phase1_default.yaml')
policy = MAPPOPolicy(cfg)
trainer = MAPPOTrainer(cfg, policy)
print('MAPPO stub OK, params:', sum(p.numel() for p in policy.parameters()))
"
```
（实际不要求 policy 有真实权重，stub 阶段只要能 instantiate 即可）

---

### Task D：MAPPO-L 算法 Stub
**负责人建议：phd2**
**输入：** `chauncygu/Multi-Agent-Constrained-Policy-Optimisation` 代码参考 + `configs/phase1_default.yaml`
**输出：** `algos/mappo_lagrangian/` 目录（含 `policy.py` 和 `trainer.py`）
**内容：**
- 与 MAPPO 的区别：增加 `cost_return`, `cumulative_cost_limit`, `lagrangian_lr` 字段
- `train()` 方法比 MAPPO 多一步 Lagrangian multiplier 更新
- **仍然只是 stub，不跑全量训练**

**验收标准：**
```
python -c "
from algos.mappo_lagrangian import MAPPOLPolicy, MAPPOLTrainer
from src.config import load_config
cfg = load_config('configs/phase1_default.yaml')
policy = MAPPOLPolicy(cfg)
trainer = MAPPOLTrainer(cfg, policy)
assert hasattr(trainer, 'update_lagrangian')
print('MAPPO-L stub OK, cost_limit:', cfg['algo']['cost_limit'])
"
```

---

### Task E：Phase1 Smoke Test（集成）
**负责人建议：phd2**
**输入：** Task A/B/C/D 的产出物
**输出：** `scripts/phase1_smoke_test.py` + 在 `results/phase1/` 下生成运行记录
**内容：**
- 用随机策略（random policy）在 SafeAnt-2x4 上跑 1 个 episode
- 不做真实训练，只验证：env → policy → step → reward/cost 收集 → logger 输出
- 记录：episode_return, episode_cost, episode_length 到 JSON 文件

**验收标准：**
```
xvfb-run -a python scripts/phase1_smoke_test.py
# 期望：脚本正常退出，打印 episode_return, episode_cost, episode_length
# results/phase1/smoke_YYYYMMDD/ 下有 run_log.json
```

---

## 4. Stub vs 复用边界

| 模块 | 复用（参考） | Stub（占位） | 说明 |
|------|-----------|------------|------|
| 环境 | Safe MAMujoco（pip 或 git clone） | — | 已有成熟实现，adapter 只是 wrapper |
| MAPPO | marlbenchmark/on-policy（参考结构） | ✅ policy/trainer stub | 不直接拷贝，保留接口干净 |
| MAPPO-L | chauncygu/MACPO（参考 Lagrangian 层） | ✅ policy/trainer stub | 只借鉴损失函数形式 |
| Config/Logger | — | ✅ 新写 | 少量代码，快速实现 |
| 训练循环 | — | ✅ stub | phase1 只跑 smoke test，不跑真实训练 |

**红线：**
- 不在 phase1 引入 MACPO / HAPPO / MATD3 / FACMAC
- 不写完整的 replay buffer / optimizer / scheduler
- 不做多 episode 并行训练

---

## 5. 任务依赖关系

```
Task A（Env Adapter）         ← 无依赖，最先
    ↓
Task B（Config System）      ← 无依赖，并行或紧随
    ↓
Task C（MAPPO Stub）         ← 依赖 Task B
Task D（MAPPO-L Stub）       ← 依赖 Task B
    ↓
Task E（Phase1 Smoke Test）  ← 依赖 Task A+B+C+D 全部
```

---

## 6. 风险与注意事项

| 风险 | 影响 | 缓解 |
|------|------|------|
| Safe MAMujoco 环境在服务器上安装失败 | Task A 阻塞 | 参考 environment-smoke-test-plan.md 的路线 A/B |
| gymnasium 与 Safe MAMujoco API 不兼容 | Task A 阻塞 | 优先用 gymnasium.make 兼容模式，必要时降级到 gym API |
| 两套 algo stub 过于复杂 | Phase1 延期 | 严格控制 stub 范围：只实现类签名，不做真实 forward/backward |
| 多人在同一 branch 冲突 | 协作阻塞 | 每人一个独立 branch（如 `feat/env-adapter`, `feat/config-system`），统一 merge 到 `develop` |

---

*本文件为 phase1 执行计划，待 Godclaw review 后冻结。*

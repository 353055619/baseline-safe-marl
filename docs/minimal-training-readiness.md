# Minimal Training Readiness Checklist

> 目标：评估 baseline-safe-marl 是否具备进入"最小训练"阶段的条件。
> ⚠️ 这是 minimal training readiness，不是 benchmark readiness。
> 最后更新：2026-03-31

---

## 一、已具备的条件

| 组件 | 状态 | 位置 |
|------|------|------|
| 6 个算法 stub | ✅ 6/6 完成 | `algos/<algo>/` |
| Stub 验收测试 | ✅ 6/6 PASS | `scripts/smoke_test_algos.py` |
| Config enrichment | ✅ 就绪 | `src/algo_config.py` |
| Fallback env adapter | ✅ 就绪 | `envs/safe_mamujoco_adapter.py` |
| Cost wrapper | ✅ 就绪 | `envs/fallback_cost_wrapper.py` |
| 1 episode demo | ✅ 存在 | `scripts/demo_episode_mappo.py` |
| smoke test | ✅ 4 envs 通过 | `results/phase1/fallback/` |

---

## 二、进入最小训练前仍需补齐的

| # | 缺口 | 说明 | 优先级 |
|---|------|------|--------|
| 1 | **统一 demo 入口脚本** | 当前 demo 是单 algo 脚本；需要一个通用入口：`python scripts/demo.py --algo MAPPO-L`，支持 YAML 指定 algo 和 env | 高 |
| 2 | **Rollout buffer（on-policy）** | MAPPO-L / MACPO / MAPPO 需要 rollout buffer 存储 (obs, action, reward, done)，支持 GAE advantage 计算 | 高 |
| 3 | **Replay buffer（off-policy）** | MATD3 / FACMAC 需要 replay buffer 支持随机采样 | 高 |
| 4 | **Episode 统计日志** | 当前 smoke test 输出到 JSON；最小训练需要统一 logger，记录每 episode 的 return / cost / length / timestep | 中 |
| 5 | **Checkpoint 保存/加载** | 训练中途需要能保存和恢复 policy 权重 | 低（可后续补） |

---

## 三、最小训练的定义

以下均**不是**最小训练的目标：
- ❌ 完整 benchmark sweep
- ❌ 超参数搜索
- ❌ 多算法对比评估
- ❌ 与其他公开 benchmark 比性能

最小训练**只要求**：
- ✅ 指定 algo + env，运行 N episodes（或 N steps）
- ✅ Policy 权重持续更新
- ✅ 每 episode 输出 return / cost / length
- ✅ 能保存最终权重

---

## 四、下一步建议

1. 先完成**统一 demo 入口**（缺口 #1），打通 config → env → policy → trainer → rollout → log 的完整链路
2. 在 demo 入口中**集成 rollout buffer**（缺口 #2，on-policy），让 MAPPO-L 能跑起来
3. Off-policy buffer（缺口 #3）可在 demo 入口稳定后单独加

---

## 五、状态判断

当前是否可以开始最小训练？

- 如接受**手动替换 algo name + hardcoded 少量参数**：`scripts/demo_episode_mappo.py` 已可演示
- 如要求**config 驱动 + 统一入口**：缺口 #1 尚未解决，需要先做统一 demo 入口

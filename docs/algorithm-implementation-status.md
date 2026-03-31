# Algorithm Implementation Status

> 面向 Godw 审阅，记录 6 个候选算法的当前实现状态。
> 最后更新：2026-03-31（smoke test 6/6 PASS）

## 总览

| 算法 | 定位 | Policy 类型 | 安全算法 | 状态 |
|------|------|------------|---------|------|
| MACPO | 基线安全 | On-policy (PPO) | ✅ | Stub 完成 |
| MAPPO-L | 安全 | On-policy (PPO) + Lagrangian | ✅ | Stub 完成 |
| MAPPO | 基线（非安全） | On-policy (PPO) | ❌ | Stub 完成 |
| HAPPO | 基线（非安全） | On-policy (PPO) + 异质 | ❌ | Stub 完成 |
| MATD3 | 基线（非安全） | Off-policy (DDPG) + Twin Critic | ❌ | Stub 完成 |
| FACMAC | 基线（非安全） | Off-policy + Factorized Critic + Mixing | ❌ | Stub 完成 |

---

## 各算法详情

### MACPO
- **目录**：`algos/macpo/`
- **安全**：✅ 是（理论每步满足约束）
- **On/Off**：On-policy
- **stub 完成**：policy.py + trainer.py
- **下一步**：验收测试（instantiate + get_actions + train loop）

### MAPPO-L
- **目录**：`algos/mappo_lagrangian/`
- **安全**：✅ 是（Lagrangian 乘子法）
- **On/Off**：On-policy
- **stub 完成**：policy.py + trainer.py（额外含 `update_lagrangian()`）
- **下一步**：与 MACPO 共用大部分逻辑，优先做 MAPPO-L 验收

### MAPPO
- **目录**：`algos/mappo/`
- **安全**：❌ 非安全基线
- **On/Off**：On-policy
- **stub 完成**：policy.py + trainer.py
- **下一步**：与 MAPPO-L 同期验收

### HAPPO
- **目录**：`algos/happo/`
- **安全**：❌ 非安全基线
- **On/Off**：On-policy（支持异质智能体）
- **stub 完成**：policy.py + trainer.py
- **下一步**：独立验收（与 MAPPO/MAPPO-L 并行）

### MATD3
- **目录**：`algos/matd3/`
- **安全**：❌ 非安全基线
- **On/Off**：Off-policy（DDPG 族）
- **stub 完成**：policy.py + trainer.py（含 twin critics + target networks + `delay` 更新）
- **下一步**：验收测试（evaluate_actions 返回 q1/q2）

### FACMAC
- **目录**：`algos/facmac/`
- **安全**：❌ 非安全基线
- **On/Off**：Off-policy
- **stub 完成**：policy.py + trainer.py（含 per-agent Q_i + QMIX-style mixing network）
- **下一步**：验收测试（evaluate_actions 返回 q_tot）

---

## 下一步优先顺序（建议）

1. ~~批量验收测试~~ ✅ 完成（Task A）：6/6 PASS，`scripts/smoke_test_algos.py`
2. ~~MAPPO-L / FACMAC / MATD3 单独验收~~ ✅ 已并入统一 smoke test
3. **与 fallback CostWrapper 集成**：algo stub + env wrapper 端到端跑通 1 episode（运行说明见 `docs/smoke-test-quickstart.md`）
4. **Config 系统接入**：`src/algo_config.py` 已就绪（详见 `docs/config-integration-notes.md`）
5. **进入最小训练前检查**：详见 `docs/minimal-training-readiness.md`

---

## 当前阻塞

- 原生 Safe MAMujoco：`mujoco_py + MuJoCo 2.1.0 binary` 仍无公开来源，fallback CostWrapper 已就绪
- Stub smoke test 已在 server-2（RTX 3090，`conda safe-marl`）完成 ✅

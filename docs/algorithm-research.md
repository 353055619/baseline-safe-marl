# Safe MARL Baseline — 算法候选调研报告

> 调研人：phd2
> 日期：2026-03-30
> 目标：为 `baseline-safe-marl` 项目筛选 6 个多智能体 Safe RL 算法，建立初版候选池

---

## 一、项目背景

- **Benchmark**: Safe Multi-Agent MuJoCo（Safe MAMujoco）
- **父项目参考**: MAMuJoCo（schroederdewitt/multiagent_mujoco，2020）
- **安全版 benchmark**: chauncygu/Safe-Multi-Agent-Mujoco（⭐73）
- **核心环境**: Ant（2x4, 3x4）、HalfCheetah（2x3, 2x6）、Hopper、Walker 等多智能体控制任务，带碰撞/跌倒约束

---

## 二、候选池（8 → 6）

### 候选 1：MACPO
| 属性 | 内容 |
|------|------|
| 全称 | Multi-Agent Constrained Policy Optimization |
| 年份 | 2021 |
| 论文 | arxiv:2110.02793 |
| 多智能体 | ✅ |
| Safe RL | ✅（理论保证每步满足约束） |
| 开源代码 | `chauncygu/Multi-Agent-Constrained-Policy-Optimisation`（⭐222） |
| Benchmark 匹配 | ✅（论文原始 benchmark 即 Safe MAMujoco） |
| 风险点 | 与 MAPPO-L 共享大量代码，实际区别在于 trust region + 约束推导；实现难度中等 |

**结论**：✅ 纳入 Shortlist。安全 MARL 领域奠基之作，理论最solid，benchmark 直接匹配。

---

### 候选 2：MAPPO-Lagrangian（MAPPO-L）
| 属性 | 内容 |
|------|------|
| 全称 | MAPPO with Lagrangian Safety Multipliers |
| 年份 | 2021 |
| 论文 | 同 MACPO（同一代码库） |
| 多智能体 | ✅ |
| Safe RL | ✅（Lagrangian 乘子法处理约束） |
| 开源代码 | 同 MACPO 仓库 |
| Benchmark 匹配 | ✅ |
| 风险点 | 与 MACPO 同源于一个代码库，复现工作量不大；两者差异在于优化框架 |

**结论**：✅ 纳入 Shortlist。与 MACPO 形成对比（不同的安全处理框架），均在 Safe MAMujoco 上验证过。

---

### 候选 3：MAPPO
| 属性 | 内容 |
|------|------|
| 全称 | Multi-Agent Proximal Policy Optimization |
| 年份 | 2021 |
| 论文 | arxiv:2103.01955 |
| 多智能体 | ✅ |
| Safe RL | ❌（无内建安全机制） |
| 开源代码 | `marlbenchmark/on-policy`（⭐1941，官方） |
| Benchmark 匹配 | ✅ |
| 风险点 | 非安全基线，用于对比；是目前最广泛使用的 MARL 基线之一，便于做 ablution |

**结论**：✅ 纳入 Shortlist。作为**非安全**基线必须纳入，用于体现安全机制的价值。

---

### 候选 4：HAPPO
| 属性 | 内容 |
|------|------|
| 全称 | Heterogeneous-Agent Proximal Policy Optimization |
| 年份 | 2021 |
| 论文 | arxiv:2109.11251 |
| 多智能体 | ✅（支持异质智能体） |
| Safe RL | ❌（无内建安全机制） |
| 开源代码 | `marlbenchmark/on-policy`（与 MAPPO 同一仓库，⭐1941） |
| Benchmark 匹配 | ✅ |
| 风险点 | 无独立安全版本；HAPPO 本身是 MAPPO 的异质版本，不带安全保证 |

**结论**：✅ 纳入 Shortlist。异质智能体场景的实际基线，与 MAPPO 互补。

---

### 候选 5：MATD3
| 属性 | 内容 |
|------|------|
| 全称 | Multi-Agent Twin Delayed DDPG |
| 年份 | 2019（MATD3）；原始 TD3 2018 |
| 论文 | MATD3: arxiv:1910.01465 |
| 多智能体 | ✅ |
| Safe RL | ❌（无内建安全机制） |
| 开源代码 | `agi-brain/xuance`（⭐1055，包含 MATD3）、`Lizhi-sjtu/MARL-code-pytorch` |
| Benchmark 匹配 | ✅ |
| 风险点 | 较老（2019），但经典；做 Off-policy 对比有用；无安全版本 |

**结论**：✅ 纳入 Shortlist。Off-policy 对比维度的代表算法，原理与 MAPPO/MACPO 完全不同，可作为对比维度。

---

### 候选 6：QMIX
| 属性 | 内容 |
|------|------|
| 全称 | QMIX: Monotonic Value Function Factorisation for Deep MARL |
| 年份 | 2018 |
| 论文 | arxiv:1806.01910 |
| 多智能体 | ✅（值分解方法） |
| Safe RL | ❌（无内建安全机制） |
| 开源代码 | PyMARL（⭐1047）、MARL-code-pytorch（⭐725）等 |
| Benchmark 匹配 | 部分（MAMujoco 上可跑，但 QMIX 主要验证于离散/SMAC 场景） |
| 风险点 | **风险较高**：QMIX 主要适用于离散动作，不符合 MuJoCo 连续控制场景；需要 actor-critic 改造；与 On-policy 方法体系不同，接入成本高 |

**结论**：⚠️ **不纳入**。与 Continuous MuJoCo benchmark 匹配度低，强行接入成本高，优先级应低于其他选项。

---

### 候选 7：FACMAC
| 属性 | 内容 |
|------|------|
| 全称 | FACTORED MULTI-Agent Centralized Policy Gradients |
| 年份 | 2022 |
| 论文 | arxiv:2201.06233 |
| 多智能体 | ✅ |
| Safe RL | ❌ |
| 开源代码 | xuance 库（⭐1055）包含该实现 |
| Benchmark 匹配 | ✅（Continuous control） |
| 风险点 | 比 MATD3/MADDPG 新，但同样无安全机制 |

**结论**：⚠️ 备选。若 MATD3 觉得太老，可以替换。

---

### 候选 8：IPPO
| 属性 | 内容 |
|------|------|
| 全称 | Independent PPO |
| 年份 | 2020 |
| 多智能体 | ✅（每个智能体独立策略） |
| Safe RL | ❌ |
| 开源代码 | marlbenchmark/on-policy（同一仓库） |
| Benchmark 匹配 | ✅ |
| 风险点 | 过于简单（独立策略，无中心化训练）；性能通常不如 MAPPO/HAPPO |

**结论**：⚠️ 备选。简单基线，但如果需要"最简 baseline" 可以纳入，但目前候选池已满。

---

## 三、推荐 Shortlist（初版 6 个）

| # | 算法 | 安全 | 主要角色 | 代码可用性 |
|---|------|------|---------|-----------|
| 1 | **MACPO** | ✅ 约束满足 | 安全方法·理论核心 | ✅ 222⭐ |
| 2 | **MAPPO-Lagrangian** | ✅ Lagrangian | 安全方法·工程友好 | ✅ 同上 |
| 3 | **MAPPO** | ❌ | 非安全基线 | ✅ 1941⭐ |
| 4 | **HAPPO** | ❌ | 异质智能体基线 | ✅ 同 MAPPO |
| 5 | **MATD3** | ❌ | Off-policy 对比 | ✅ xuance 1055⭐ |
| 6 | **FACMAC** | ❌ | 2022新版·值分解对比 | ✅ xuance 1055⭐ |

**说明**：纳入逻辑
- 安全维度（2个）：MACPO + MAPPO-L = 两种不同的约束处理框架
- 非安全基线（4个）：MAPPO + HAPPO（on-policy）+ MATD3 + FACMAC（off-policy/value decomposition）
- QMIX 因与 Continuous MuJoCo 不匹配而排除

---

## 四、风险分析

| 风险 | 影响算法 | 建议 |
|------|---------|------|
| QMIX 不适合 continuous MuJoCo | QMIX | 已排除 |
| FACMAC/MATD3 无内建安全，需二次开发安全层 | MATD3, FACMAC | 保持非安全基线定位，不过度承诺 |
| MACPO 与 MAPPO-L 代码高度重叠 | MACPO, MAPPO-L | 第一阶段优先实现 MAPPO-L（更简单），再迁移到 MACPO |
| HAPPO 无独立安全版本 | HAPPO | 不强求安全版，以异质基线角色纳入 |
| 6 个算法工程量较大 | 全部 | SDD+小步快跑，第一阶段先跑通 1-2 个 |

---

## 五、第一阶段优先原型推荐

**推荐优先接入：MAPPO-Lagrangian（MAPPO-L）**

理由：
1. **Benchmark 直接匹配** — 同一作者群已在 Safe MAMujoco 上验证，拿来即用
2. **代码已开源** — `chauncygu/Multi-Agent-Constrained-Policy-Optimisation` 完整可跑
3. **概念直观** — 在 MAPPO 基础上加 Lagrangian 乘子，工程理解成本低
4. **团队练手价值高** — 把 MAPPO-L 跑通后，MACPO 的接入成本大幅降低（共享 80% 代码）
5. **快速产出** — 第一阶段就能展示安全 baseline 效果，给 Godw 可见的初期成果

**第二优先级**：MAPPO（非安全版，作为对照组快速上线）
**第三优先级**：MACPO（理论最solid，但与 MAPPO-L 重复工作量大）

---

## 六、还需进一步确认的事项

- [ ] benchmark 环境的 gymnasium 版本兼容性（Safe MAMujoco 最新版依赖）
- [ ] on-policy 代码库是否需要额外改装以适配 Safe MAMujoco
- [ ] 各算法在 multi-agent safe mujoco 上的公开 benchmark 数据（用于对比参考）
- [ ] xuance 库是否已包含所有目标算法（避免重复造轮子）
- [ ] HPC 集群/GPU 环境是否支持 mujoco-py（需要 X server 或 offscreen rendering）

---

*本报告为初版候选筛选，待 Godclaw review 后调整。*

> 📌 stub 实现状态已更新 → `docs/algorithm-implementation-status.md`

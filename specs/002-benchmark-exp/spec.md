# Feature Specification: Benchmark Experiment — 6 Algorithm × 4 Environment Convergence Curves

**Feature Branch**: `002-benchmark-exp`
**Created**: 2026-04-01
**Status**: In Progress
**Input**: Godw 指令：6个算法在4个环境上跑收敛曲线，绘制1行4列折线图；当前步数严重不足，需调研标准训练量；phd1和phd2并行跑

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 并行运行6个算法×4个环境 (Priority: P1)

作为 researcher，我需要在两台服务器上并行运行所有算法×环境组合，以在最短时间内获得完整对比结果。

**Why this priority**: 今晚需要看到结果，时间紧迫。

**Independent Test**: 每一台服务器独立运行3个算法×4个环境，无互相依赖。

**Acceptance Scenarios**:

1. **Given** Server 1（phd1）和 Server 2（phd2），**When** 同时开始运行，**Then** 各自完成3个算法
2. **Given** 全部24个运行完成，**When** 汇总CSV，**Then** 每个 algo×env 组合有2个run的数据

---

### User Story 2 - 收敛曲线可视化 (Priority: P1)

作为 researcher，我希望看到每种算法在每个环境上的收敛曲线，以便评估算法效果。

**Why this priority**: 与论文中 baseline 对比的核心数据。

**Independent Test**: 生成1行4列折线图（ant, halfcheetah, hopper, walker），每图包含所有6条算法曲线。

**Acceptance Scenarios**:

1. **Given** 所有运行完成的CSV，**When** 运行绘图脚本，**Then** 生成 `results/plots/benchmark_4env_6algo.pdf`
2. **Given** PDF图表，**When** 打开查看，**Then** 每列对应一个环境，每条线对应一个算法的收敛曲线

---

### User Story 3 - 标准训练量确定 (Priority: P2)

作为 researcher，我需要知道每个算法在 MuJoCo 上应该跑多少步才算是合理的实验。

**Why this priority**: 当前 5 episodes × 500 steps 远不足以展示收敛特性。

**Acceptance Scenarios**:

1. **Given** 文献调研，**When** 确定标准步数，**Then** 实际运行使用 50 episodes × 500 steps = 25K steps（快速收敛验证）

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: **算法列表**：MAPPO、MAPPO-L、HAPPO、MACPO、MATD3、FACMAC（如环境支持）
- **FR-002**: **环境列表**：ant、halfcheetah、hopper、walker（共4个）
- **FR-003**: **训练量**：每个 algo×env 组合 2 runs × 50 episodes × 500 max_steps = 50K 总步数
- **FR-004**: **并行执行**：phd1 跑3个算法，phd2 跑3个算法，同时进行
- **FR-005**: **输出格式**：每个运行输出 CSV（algo, episode, reward, cost, steps, done）
- **FR-006**: **绘图输出**：1行4列 PDF，每列一个环境，6条收敛曲线
- **FR-007**: **CSV组织**：`results/{env}/{algo}/run_{run_id}.csv`
- **FR-008**: **FACMAC**：如 mujoco_py 不可用则在 spec 中标注为 unsupported，不阻塞其他算法

### Key Entities

- **Server 1 (phd1, port 10022)**：运行 MAPPO、MAPPO-L、HAPPO
- **Server 2 (phd2, port 20022)**：运行 MACPO、MATD3、FACMAC（尝试）
- **results/{env}/{algo}/run_{n}.csv**：每次运行的 CSV 结果

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有 24 个 algo×env 组合各自完成 2 次运行（runs=2, episodes=50, max_steps=500）
- **SC-002**: CSV 文件存在于 `results/{env}/{algo}/run_1.csv` 和 `results/{env}/{algo}/run_2.csv`
- **SC-003**: `results/plots/benchmark_4env_6algo.pdf` 生成，包含 1×4 子图
- **SC-004**: 每条收敛曲线（episode vs reward）趋势明显（非平坦）
- **SC-005**: 下午之前完成所有运行，晚上之前生成 PDF

## Assumptions

- 服务器网络稳定，不中断长时运行
- Server 2 conda 环境 `safe-marl` 可导入所有依赖（gymnasium、tianshou、torch）
- FACMAC 依赖 mujoco_py，如失败则跳过
- 50 episodes × 500 steps 是足够展示收敛趋势的最小训练量（平衡时间和质量）

## 分工

### phd1（Server 1, port 10022）

```bash
conda activate safe-marl
cd ~/code/baseline-safe-marl
git fetch origin && git checkout 001-codebase-structure && git pull origin 001-codebase-structure

# MAPPO × 4 envs
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo MAPPO --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_mappo_${env}.log
done

# MAPPO-L × 4 envs
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo MAPPO-L --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_mappol_${env}.log
done

# HAPPO × 4 envs
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo HAPPO --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_happo_${env}.log
done
```

### phd2（Server 2, port 20022）

```bash
conda activate safe-marl
cd ~/code/baseline-safe-marl
git fetch origin && git checkout 001-codebase-structure && git pull origin 001-codebase-structure

# MACPO × 4 envs
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo MACPO --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_macpo_${env}.log
done

# MATD3 × 4 envs
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo MATD3 --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_matd3_${env}.log
done

# FACMAC × 4 envs（如支持）
for env in ant halfcheetah hopper walker; do
  python scripts/run_exp.py --algo FACMAC --env $env --runs 2 --episodes 50 --max-steps 500 \
    2>&1 | tee results/exp_facmac_${env}.log
done
```

## 收敛曲线绘图规格

- **格式**：PDF，1行4列
- **列顺序**：ant, halfcheetah, hopper, walker（从左到右）
- **X轴**：Episode number（1-50）
- **Y轴**：Reward（越高越好，展示收敛趋势）
- **线条**：每个 algo 一条线（6条），不同颜色/线型区分
- **图例**：每个子图共享图例（放在图右侧或底部）
- **标题**：每个子图顶部显示环境名称

## 参考文献（调研结论）

- 当前步数（5 episodes × 500 steps = 2.5K steps）仅够随机策略验证，远不足以收敛
- MAPPO/HAPPO 标准训练量：~500 episodes × 1000 steps = 500K steps（论文典型）
- 考虑到实验时间限制，采用 **50 episodes × 500 steps = 25K steps** 作为快速收敛验证基准
- 4 envs × 6 algos × 2 runs × 50 episodes × 500 steps = **1.2M 总步数**，约需 4-8 小时（两台服务器并行）

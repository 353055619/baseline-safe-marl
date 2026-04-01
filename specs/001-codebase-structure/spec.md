# Feature Specification: baseline-safe-marl Codebase Restructure

**Feature Branch**: `001-codebase-structure`
**Created**: 2026-04-01
**Status**: Draft
**Input**: Godw 指令：参考 agv 项目结构，优化 baseline-safe-marl 目录结构；所有项目py文件放入 `src/baseline_safe_marl/`；非项目脚本删除；历史CSV归档；严格遵循 spec-kit 开发规范

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 代码库结构重组 (Priority: P1)

作为 researcher，我希望 baseline-safe-marl 代码库结构清晰、可维护，以便团队成员能快速定位代码、协作开发。

**Why this priority**: 代码库结构是一切开发效率的基础，当前散乱的目录严重阻碍日常工作。

**Independent Test**: 重构完成后，运行 `python -c "from baseline_safe_marl import algos, envs; print('import OK')"` 验证所有模块可正常导入。

**Acceptance Scenarios**:

1. **Given** 新的 `src/baseline_safe_marl/` 目录结构，**When** 执行 `python -m baseline_safe_marl --help`，**Then** 显示正确帮助信息
2. **Given** 重组后的代码库，**When** 从 Server 2 运行 `python scripts/run_exp.py --algo MATD3 --env ant --episodes 2`，**Then** 成功输出 CSV 结果
3. **Given** 所有算法代码在 `src/baseline_safe_marl/algos/` 下，**When** 查看目录结构，**Then** on_policy 和 off_policy 分类清晰

---

### User Story 2 - 实验可复现运行 (Priority: P1)

作为 researcher，我希望所有算法能在统一入口运行并输出可比结果，以便快速验证算法效果。

**Why this priority**: 今晚需要看到所有算法的收敛曲线，时间紧迫。

**Independent Test**: 重构后，从 Server 2 运行 `python scripts/run_exp.py --algo MAPPO --env ant --runs 2 --episodes 10`，生成 CSV 和 TensorBoard 日志。

**Acceptance Scenarios**:

1. **Given** `scripts/run_exp.py` 统一入口，**When** 指定 `--algo HAPPO --env ant`，**Then** 正确加载 HAPPO 算法并运行
2. **Given** 所有算法运行完成，**When** 查看 `results/` 目录，**Then** 每个算法有独立 CSV 文件

---

### User Story 3 - 历史文件归档 (Priority: P2)

作为 researcher，我希望历史实验数据被归档而不是删除，以便追溯早期实验结果。

**Why this priority**: 历史数据有参考价值，不能直接丢弃。

**Independent Test**: 归档后，`results/archive/` 目录包含所有历史 CSV 文件，且 `results/` 根目录不再有散落的 CSV。

**Acceptance Scenarios**:

1. **Given** 历史 CSV 文件（`FACMAC.csv`、`HAPPO.csv` 等），**When** 重构执行，**Then** 全部移入 `results/archive/`
2. **Given** `results/archive/` 目录，**Then** 包含所有历史 CSV 且命名不变

---

### User Story 4 - 配置体系重建 (Priority: P2)

作为 researcher，我希望有分层、DRY 的配置系统，以便高效管理多实验多配置。

**Why this priority**: 当前仅1个 YAML 无法支撑后续大规模实验。

**Independent Test**: 重构后，`configs/exp/exp1_onpolicy_safe.yaml` 能引用 `configs/defaults/phase1_default.yaml` 作为基础配置。

**Acceptance Scenarios**:

1. **Given** `configs/defaults/phase1_default.yaml`，**When** 新实验配置引用它，**Then** 共享配置自动合并
2. **Given** `configs/algos/matd3.yaml`，**When** 运行 MATD3 实验，**Then** 使用 MATD3 特定超参数

---

### User Story 5 - Star Office UI 脚本清理 (Priority: P3)

作为 researcher，我希望与项目无关的脚本被清理，以便代码库保持干净。

**Why this priority**: `set_state.py`、`office-agent-push.py` 与 Safe MARL 研究无关。

**Acceptance Scenarios**:

1. **Given** 项目根目录和 `scripts/` 目录，**When** 重构完成，**Then** 不存在 `set_state.py`、`office-agent-push.py`、`demo_episode.py`

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 所有项目相关 `.py` 文件必须位于 `src/baseline_safe_marl/` 目录下
- **FR-002**: `src/baseline_safe_marl/` 必须是一个合法的 Python 包（包含 `__init__.py` 和 `__main__.py`）
- **FR-003**: 所有算法入口统一为 `scripts/run_exp.py`，用法：`python scripts/run_exp.py --algo <ALGO> --env <ENV> --runs <N> --episodes <M>`
- **FR-004**: `.venv` 虚拟环境必须在项目根目录下（从 `results/.venv/` 迁移）
- **FR-005**: 所有历史 CSV 文件归档到 `results/archive/`
- **FR-006**: `third_party/`、`feature/` 目录删除；Star Office UI 推送脚本删除
- **FR-007**: `configs/` 重建为三层结构：`defaults/`、`algos/`、`envs/`、`exp/`
- **FR-008**: `docs/` 按 `architecture/`、`algorithms/`、`specs/`、`api/` 分层重组
- **FR-009**: 重构完成后，所有 5 个算法（MAPPO、MAPPO-L、HAPPO、MACPO、MATD3）在至少 1 个环境（ant）上能成功运行并输出结果
- **FR-010**: FACMAC 算法如服务器环境支持则包含，否则在 spec 中标注为待支持

### Key Entities

- **src/baseline_safe_marl/algos/on_policy/**: On-policy 算法子包（MAPP O、HAPPO、MACPO）
- **src/baseline_safe_marl/algos/off_policy/**: Off-policy 算法子包（MATD3、FACMAC）
- **src/baseline_safe_marl/envs/core/**: Fallback 环境实现（Adapter + CostWrapper）
- **scripts/run_exp.py**: 统一实验入口脚本
- **configs/exp/**: 实验配置目录

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `python -c "from baseline_safe_marl import algos, envs; print('OK')"` 执行成功，退出码 0
- **SC-002**: `python scripts/run_exp.py --algo MATD3 --env ant --runs 1 --episodes 3` 在 Server 2 上成功完成，生成 `results/exp1_onpolicy_safe/matd3/matd3.csv`
- **SC-003**: 所有 5 个算法（MAPPO、MAPPO-L、HAPPO、MACPO、MATD3）在 ant 环境上各自完成至少 3 episodes，生成收敛曲线数据
- **SC-004**: `results/` 目录中无散落 CSV；所有 CSV 在 `results/archive/` 或 `results/{exp_name}/{algo}/`
- **SC-005**: `git status` 显示干净的 working directory（无无关文件），branch `001-codebase-structure` 包含完整重组结果

## Assumptions

- 服务器环境（Server 2，Python 3.10，conda `safe-marl`）可用于运行实验
- FACMAC 需要 `mujoco_py` 支持，如不可用则标记为待支持，不阻塞主流程
- 重构期间 phd2 可同时进行实验（不影响重构进度）
- spec-kit 工具已安装（`uv tool install specify-cli`），所有 agent 严格按 constitution 执行

## Migration Tasks *(detailed)*

### Phase 1: 骨架搭建（phd1 负责）
1. 创建 `src/baseline_safe_marl/` 目录结构（algos/on_policy/、algos/off_policy/、envs/core/、envs/native/）
2. 移动 `algos/` → `src/baseline_safe_marl/algos/`
3. 移动 `envs/` → `src/baseline_safe_marl/envs/`，拆分为 `core/` + `native/`
4. 移动 `src/config.py`、`src/algo_config.py` → `src/baseline_safe_marl/`
5. 新增 `src/baseline_safe_marl/logger.py`、`registry.py`、`env_utils.py`、`constants.py`
6. 创建 `src/baseline_safe_marl/__init__.py` 和 `__main__.py`

### Phase 2: 入口收拢与清理（phd1 负责）
7. 创建 `scripts/run_exp.py` 统一入口
8. 移动所有根目录 `.py` 入口到 `scripts/`
9. 删除：`set_state.py`、`office-agent-push.py`、`demo_episode.py`（未patch）、`office-agent-state.json`
10. 迁移 `results/.venv/` → 根目录 `.venv/`

### Phase 3: 归档与配置（phd1 负责）
11. 创建 `results/archive/`，移动所有历史 CSV
12. 删除 `feature/`、`third_party/`
13. 创建 `configs/defaults/`、`configs/algos/`、`configs/envs/`、`configs/exp/` 目录结构

### Phase 4: 实验运行（phd2 负责，与 Phase 1-3 并行）
14. Server 2 环境验证：`conda activate safe-marl && python -c "import gymnasium, tianshou; print('OK')"`
15. 运行所有 5 个算法在 ant 环境上的收敛实验（每个 10+ episodes）
16. 生成 CSV 结果和收敛曲线

# Environment Smoke Test Plan

> 生成时间：2026-03-30
> 作者：phd1
> 目标：多智能体 Safe RL baseline — 第一阶段最小可运行 demo
> 状态：**部分验证（2026-03-30 server-2 实测）**

---

## 1. 背景

当前项目目标：6 个多智能体 Safe RL 算法的 baseline 实现，Benchmark 为 multi-agent safe mujoco。
本阶段目标：**仅验证环境可跑，不做训练**。

---

## 2. 服务器环境现状（已验证）

| 组件 | 状态 | 命令验证 |
|---|---|---|
| Python 3.14.3 | ✅ | `conda activate agv && python --version` |
| gymnasium 1.2.3 | ✅ | `python -c 'import gymnasium; print(gymnasium.__version__)'` |
| mujoco (Python) | ❌ 未安装 | `python -c 'import mujoco'` → ModuleNotFoundError |
| mujoco_py | ❌ 未安装 | pip list 中无 mujoco 相关 |
| ~/.mujoco/ 目录 | ❌ 不存在 | `ls ~/.mujoco/` → No such file |
| xvfb-run | ✅ 可用 | `which xvfb-run` |
| DISPLAY | ❌ 未设置 | 服务器无 X，需 xvfb |

---

## 3. 两条路线对比

| | 路线 A（推荐）新栈 | 路线 B 旧栈 |
|---|---|---|
| 核心依赖 | `pip install mujoco` | `pip install mujoco-py==2.0.2.8 gym==0.17.2` |
| gym 版本 | gymnasium 1.2.3（已有） | gym 0.17（需额外装） |
| mujoco binary | pip 自动带 | 手动下载到 ~/.mujoco/ |
| 渲染 | xvfb-run | xvfb-run + LD_PRELOAD |
| 复杂度 | **低** | 高 |
| 算法兼容性 | 待测 | MACPO/MAPPO-L 原始代码需要此栈 |
| 预计阻塞时间 | **< 1 小时** | 可能 1-2 天 |

---

## 4. 推荐路线 A 执行步骤

### Step 1：在服务器上安装 mujoco Python 包

```bash
ssh godw@172.20.135.15 -p 10022
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agv
pip install mujoco
```

### Step 2：在服务器上写最小 demo

在 `~/code/phd1/` 新建 `safe_mujoco_demo.py`：

```python
"""
最小 smoke test：multi-agent safe mujoco 环境 reset / step / rollout
不训练，只验证环境可运行
"""
import gymnasium as gym

# 验证 gymnasium + mujoco 基础环境（单智能体 mujoco 用作 sanity check）
def test_single_mujoco():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    total_reward = 0
    for step_i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset(seed=42)
    env.close()
    print(f"[OK] HalfCheetah-v5: 100 steps, total_reward={total_reward:.2f}")
    return True

# 验证 multi-agent 场景（如果 gymnasium 支持 multi-agent mujoco）
def test_multi_agent_mujoco():
    # gymnasium 没有原生 multi-agent safe mujoco，
    # 这里先用 PettingZoo MARL 环境做验证
    # 实际 Safe MAMujoco 需要 chauncygu/Safe-Multi-Agent-Mujoco
    try:
        from pettingzoo.mujoco import multiagent_pendulum_v1
        env = multiagent_pendulum_v1.parallel_env(render_mode="rgb_array")
        observations, infos = env.reset(seed=42)
        for step_i in range(50):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            if all(terminations.values()) or all(truncations.values()):
                observations, infos = env.reset(seed=42)
        env.close()
        print(f"[OK] multiagent_pendulum_v1: 50 steps")
        return True
    except Exception as e:
        print(f"[FAIL] multiagent_pendulum_v1: {e}")
        return False

if __name__ == "__main__":
    print("=== baseline-safe-marl smoke test ===")
    ok1 = test_single_mujoco()
    ok2 = test_multi_agent_mujoco()
    if ok1 and ok2:
        print("=== ALL PASSED ===")
    else:
        print("=== PARTIAL FAILURE ===")
```

### Step 3：用 xvfb 跑 demo

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agv
xvfb-run -a --server-args="-screen 0 1024x768x24" \
    python ~/code/phd1/safe_mujoco_demo.py
```

---

## 5. 路线 B 步骤（备选，仅在算法确实需要时）

```bash
# Step 1: 下载 mujoco binary
mkdir -p ~/.mujoco/mujoco200/bin
cd ~/.mujoco/mujoco200/bin
wget https://www.roboti.us/file/get?file=mujoco200_linux.zip
unzip mujoco200_linux.zip

# Step 2: 安装旧栈
pip install gym==0.17.2 mujoco-py==2.0.2.8

# Step 3: 设置环境变量
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Step 4: 验证
python -c 'import mujoco_py; print("mujoco_py OK")'
```

---

## 6. 成功标准

| 测试 | 预期 |
|---|---|
| `import mujoco` | 无报错 |
| `HalfCheetah-v5 reset/step` | 正常运行 100 steps |
| `multiagent_pendulum_v1 reset/step` | 正常运行 50 steps |
| xvfb 下渲染不崩 | 能生成 rgb_array 或 headless 模式正常 |

---

## 7. 阻塞升级条件

如果以下任一情况发生且 30 分钟内无法自行解决，升级为 blocker 写入 drafts：

- [ ] `pip install mujoco` 报错（网络/权限/依赖问题）
- [ ] `import mujoco` 成功但 `env.reset()` 报 GLX 错误
- [ ] xvfb-run 启动失败
- [ ] 找不到 gymnasium multi-agent 环境

---

*本文档为第一阶段 smoke test 执行计划，待环境验证后更新状态。*

---

## 8. 实测结果（2026-03-30，server-2 RTX 3090）

### 环境配置
```bash
conda create -n safe-marl python=3.10 -y
conda activate safe-marl
pip install mujoco gymnasium[mujoco] pettingzoo  # mujoco-3.6.0, gymnasium-1.2.3, pettingzoo-1.25.0
pip install gym  # gym-0.26.2 (for Safe MAMujoco compatibility)
pip install jinja2  # Safe MAMujoco dependency
```

### 测试结果

| 测试 | 环境 | 结果 | 备注 |
|---|---|---|---|
| `gymnasium.make("HalfCheetah-v5")` | safe-marl / Python 3.10 | ✅ **PASS** | 100 steps, reward=-5.22 |
| Safe MAMujoco ant | safe-marl / Python 3.10 | ❌ **BLOCKED** | 缺 `mujoco210/bin/mujoco` |
| mujoco_py import | safe-marl / Python 3.10 | ❌ **BLOCKED** | 缺 MuJoCo 2.1.0 binary |
| pettingzoo MARL | safe-marl / Python 3.10 | ❌ **BLOCKED** | pettingzoo 1.25 无 mujoco 子模块 |

### 关键发现
- **mujoco 3.6.0（gymnasium 新栈）在 GPU 上完全可用** ✅
- Safe MAMujoco 依赖 `mujoco_py` + `gym==0.17`，需要 MuJoCo 2.1.0 binary
- `~/.mujoco/mujoco210/bin/mujoco` 不存在（2.3.1 的 bin/ 目录只有工具，无主 binary）
- DeepMind mujoco 2.3.1 没有 MuJoCo 主 binary（`bin/mujoco`）

### 下一步（优先选一）
1. 下载 MuJoCo 2.1.0 binary（从 openai/mujoco-py releases 找备用 URL）
2. 用 gymnasium Ant-v5 + 自定义 cost wrapper 做"简版 Safe MAMujoco"
3. 询问 Godw 是否有 roboti.us license 或 MuJoCo license 文件

---

## 9. Fallback Task 1 验证结果（2026-03-30，server-2 RTX 3090）

### 产物
- 文件：`envs/fallback_cost_wrapper.py` ✅
- 测试：`tests/` 或直接运行 `python envs/fallback_cost_wrapper.py`

### 验收标准（来自 phase1-fallback-plan.md）
```bash
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

### 实测结果
```
=== CostWrapper smoke test ===
[PASS] HalfCheetah + CostWrapper: 100 steps OK
[PASS] Ant + CostWrapper: 100 steps OK
=== smoke test complete ===
```
✅ **两个测试均通过**

### Cost 规则说明

| 环境 | Cost 触发条件 | 规则描述 |
|---|---|---|
| HalfCheetah-v5 | `abs(x_velocity) > 3.0 m/s` | 速度超阈值记 cost=1 |
| Ant-v5 | `torso_z < 0.25` 或 `abs(torso_y) > 5.0` | 摔倒或越出走廊边界记 cost=1 |
| Hopper | `||action||_2 > 10.0` | 动作幅度过大（不稳定代理）记 cost=1 |
| 默认/其他 | 无 | 暂不施加约束，cost=0.0 |

### 局限性声明（⚠️）
- 这些 cost 函数是基于经验阈值的人为规则，非物理仿真结果
- 与原生 Safe MAMujoco 的物理引擎 cost 信号完全不同
- 本 CostWrapper 仅为验证集成链路，不代表安全 benchmark 性能


---

## 10. mujoco_py 原生阻塞探查结论（server-2, 2026-03-30）

### 核心结论
**原生 Safe MAMujoco（chauncygu）卡在 mujoco_py 无法 import，原因有三层叠加：**

| 阻塞层 | 描述 | 状态 |
|---|---|---|
| 1 | MuJoCo 2.1.0 binary 无公开 x86_64 来源 | ❌ |
| 2 | Cython 3.x 与 mujoco_py Cython 代码不兼容 | ⚠️ 可降级 |
| 3 | mujoco_py 需从源码编译但构建环境难打通 | ⚠️ 需手动配置 |

### 根因
- `mujoco_py` 已停止维护（最后版本 2.1.2.14）
- DeepMind mujoco 2.2.0+ 的 SONAME 与 2.1.0 不兼容
- mujoco_py wheel 是纯 Python，不含编译扩展

### 建议
继续走 **Fallback 路线**（gymnasium + CostWrapper），不追 mujoco_py 原生路线

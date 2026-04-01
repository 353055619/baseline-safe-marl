# Config Integration Notes

> 记录 config 系统如何对接各算法 stub。面向维护者。
> 最后更新：2026-03-31

---

## Config 入口

```
configs/phase1_default.yaml   ← 统一 YAML 配置文件
src/config.py                 ← load_config()，支持 CLI override
src/algo_config.py            ← make_algo_config()，算法配置注入
```

标准用法：
```python
from src.config import load_config
from src.algo_config import make_algo_config

base = load_config("configs/phase1_default.yaml")
cfg = make_algo_config("MAPPO-L", base)  # 注入 algo-specific 字段
```

---

## `src/algo_config.py` 解决了什么问题

MAPPO/MAPPO-L 的 YAML 已含所有必要字段。但 MATD3/FACMAC 需要额外 off-policy 字段（`tau`、`policy_delay`、`exploration_noise`、`critic_lr`），而 YAML 不想为每个算法单独维护多份。

`make_algo_config(algo_name, base_cfg)` 在加载时注入缺失字段，stub 代码不变：

| 算法 | 行为 |
|------|------|
| MAPPO / MAPPO-L / HAPPO / MACPO | YAML 完整，直接透传，无注入 |
| MATD3 | 注入 off-policy 字段（tau, policy_delay, exploration_noise, critic_lr） |
| FACMAC | 注入 MATD3 字段 + mixing_hidden_dim |

---

## 哪些字段已存在，哪些需要注入

### 已存在于 `configs/phase1_default.yaml`

```yaml
algo:
  algo_name, seed, lr, gamma, lam, clip_eps, target_kl
  hidden_dim, activation, num_steps, num_epochs, batch_size
  entropy_coef, value_coef, max_grad_norm
  cost_limit, lagrangian_lr, initial_lagrangian_multiplier  # MAPPO-L 专用
  agent_type, share_policy, device
env:
  env_name, fallback, render_mode, num_envs, max_episode_steps
```

### 需要注入（MAPPO/MAPPO-L 不需要）

```python
_OFF_POLICY_DEFAULTS = {
    "tau": 0.005,              # Polyak 平滑系数
    "policy_delay": 2,         # Actor 更新频率（步）
    "exploration_noise": 0.1,  # 探索噪声
    "critic_lr": 1e-3,        # Critic 学习率
}

_FACMAC_EXTRA_DEFAULTS = {
    "mixing_hidden_dim": 64,   # Mixing 网络隐层维度
}
```

---

## 如何服务后续 demo / 最小训练

所有 stub trainer/policy 从 `cfg['algo']` 读参。只要入口层保证字段完整，demo / 最小训练脚本无需感知具体算法差异：

```python
# 通用 demo 脚本（不写死算法）
from src.config import load_config
from src.algo_config import make_algo_config

cfg = make_algo_config(cfg["algo"]["algo_name"], load_config())
policy = POLICY_MAP[cfg["algo"]["algo_name"]](cfg)
trainer = TRAINER_MAP[cfg["algo"]["algo_name"]](cfg, policy)
# ... rollout loop
```

---

## 扩展新算法时

1. 在 `src/algo_config.py` 的 `_OFF_POLICY_DEFAULTS` 或 `_FACMAC_EXTRA_DEFAULTS` 中补充缺失字段
2. 在 `make_algo_config()` 的对应 `if` 分支添加注入
3. stub 测试确保 enriched cfg 可正常 instantiate

"""
src/algo_config.py — Algorithm-Specific Config Enrichment
=========================================================
提供 algo-specific 默认字段注入，使同一 YAML 配置可以为不同算法所用。

用法:
    from src.algo_config import make_algo_config
    from src.config import load_config

    base_cfg = load_config("configs/phase1_default.yaml")
    cfg = make_algo_config("MAPPO-L", base_cfg)
    # MAPPO/MAPPO-L: 已在 YAML 完整，直接透传
    # MATD3: 注入 tau, policy_delay, exploration_noise, critic_lr
    # FACMAC: 注入 MATD3 字段 + mixing_hidden_dim
    # HAPPO: 同 MAPPO

每个 algo stub 的 trainer/policy 只需从 cfg['algo'] 读取字段，
注入在 config 层面完成，stub 代码不变。
"""

from __future__ import annotations

from typing import Any, Dict


# --------------------------------------------------------------------
# Per-algorithm default overrides
# --------------------------------------------------------------------
_OFF_POLICY_DEFAULTS: Dict[str, Any] = {
    # MATD3 / FACMAC 共享的 off-policy 字段
    "tau": 0.005,                 # Polyak smoothing coefficient
    "policy_delay": 2,            # Actor update frequency (steps)
    "exploration_noise": 0.1,     # Gaussian noise std for exploration
    "critic_lr": 1e-3,           # Critic learning rate (can differ from actor lr)
}

_FACMAC_EXTRA_DEFAULTS: Dict[str, Any] = {
    # FACMAC 额外字段
    "mixing_hidden_dim": 64,      # Mixing network hidden dim
}

# MAPPO / HAPPO 不需要额外注入（已在 YAML 完整）
# MACPO 同 MAPPO


def make_algo_config(algo_name: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于 base_cfg 注入 algo-specific 默认字段，返回完整 cfg 副本。

    参数:
        algo_name: 算法名，如 "MAPPO-L", "MATD3", "FACMAC", "MAPPO", "HAPPO", "MACPO"
        base_cfg: load_config() 返回的原始配置

    返回:
         enriched 配置字典（副本，不修改 base_cfg）
    """
    cfg = {**base_cfg}                   # shallow copy 顶层
    cfg["algo"] = {**base_cfg.get("algo", {})}  # 复制 algo section

    algo_lower = algo_name.lower()

    # --- Off-policy algorithms ---
    if algo_lower in ("matd3", "facmac"):
        cfg["algo"].update(_OFF_POLICY_DEFAULTS)

    # --- FACMAC extra ---
    if algo_lower == "facmac":
        cfg["algo"].update(_FACMAC_EXTRA_DEFAULTS)

    # No changes needed for MAPPO, MAPPO-L, HAPPO, MACPO
    # (all fields already present in YAML)

    return cfg


# --------------------------------------------------------------------
# Convenience: get canonical algo name from yaml config
# --------------------------------------------------------------------
def resolve_algo_name(cfg: Dict[str, Any]) -> str:
    """从 cfg 中提取 algo_name，规范化大小写"""
    return cfg.get("algo", {}).get("algo_name", "MAPPO-L")


# --------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import load_config

    base = load_config("configs/phase1_default.yaml", silent=True)

    for name in ["MAPPO-L", "MAPPO", "HAPPO", "MACPO", "MATD3", "FACMAC"]:
        enriched = make_algo_config(name, base)
        algo = enriched["algo"]
        off_policy_keys = ["tau", "policy_delay", "exploration_noise", "critic_lr"]
        facmac_keys = off_policy_keys + ["mixing_hidden_dim"]

        is_off = name.upper() in ("MATD3", "FACMAC")
        is_fac = name.upper() == "FACMAC"

        missing = [k for k in (facmac_keys if is_fac else off_policy_keys if is_off else [])
                   if k not in algo]

        print(f"{name:10s}: is_off={is_off}, is_fac={is_fac}, missing_after_enrich={missing} {'' if missing else '✅'}")

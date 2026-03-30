"""
src/config.py — 配置加载器
===========================
提供 load_config() 函数，支持：
  1. 从 YAML 文件加载默认配置
  2. 从 CLI arguments 覆盖配置（--config.key=value 格式）
  3. 环境变量覆盖（CONFIG_KEY=VALUE 格式）

用法:
    from src.config import load_config
    cfg = load_config("configs/phase1_default.yaml")

    # CLI override 示例:
    #   python script.py --config.algo.lr=5e-4 --config.algo.device=cuda
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _get_default_config_path() -> str:
    """自动查找 configs/phase1_default.yaml"""
    candidates = [
        Path("configs/phase1_default.yaml"),
        Path(__file__).parent.parent / "configs" / "phase1_default.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"phase1_default.yaml not found in {[str(p) for p in candidates]}. "
        "Please provide config path explicitly."
    )


def _is_float_string(s: str) -> bool:
    """
    判断字符串是否为合法浮点数表示（支持科学计数法如 5e-4, 3.14e+10）。
    """
    import re
    return bool(re.fullmatch(r'-?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?', s))


def _parse_override(s: str) -> tuple[str, Any]:
    """
    将 'algo.lr=3e-4' 解析为 ('algo.lr', 3e-4)
    支持 int/float/bool/str，自动推断类型
    """
    if "=" not in s:
        raise ValueError(f"Invalid override format: {s}, expected key=value")

    key_path, raw_val = s.split("=", 1)
    key_path = key_path.strip()
    raw_val = raw_val.strip()

    # 自动类型推断
    if raw_val.lower() == "true":
        val: Any = True
    elif raw_val.lower() == "false":
        val = False
    elif raw_val.lower() == "none":
        val = None
    elif _is_float_string(raw_val):
        # 浮点数（包括科学计数法如 5e-4, 3.14, 1e10）
        val = float(raw_val)
    elif raw_val.isdigit():
        val = int(raw_val)
    else:
        val = raw_val

    return key_path, val


def _set_nested(cfg: Dict, key_path: str, value: Any) -> None:
    """将 key_path='a.b.c' 展开为 cfg['a']['b']['c'] = value"""
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _register_cli_args(parser: argparse.ArgumentParser) -> None:
    """注册 --config.key=value 风格的 CLI 参数（不在 help 里显示，避免太长）"""
    parser.add_argument(
        "--config",
        action="append",  # 可多次使用: --config.a=1 --config.b=2
        dest="config_overrides",
        metavar="KEY=VALUE",
        help="Override config key. Example: --config.algo.lr=5e-4",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        dest="config_file",
        metavar="PATH",
        help="Path to YAML config file (default: configs/phase1_default.yaml)",
    )


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[list[str]] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    加载配置。

    参数:
        config_path: YAML 配置文件路径。
                    默认为 configs/phase1_default.yaml（相对于 cwd 或项目根目录）。
        cli_overrides: CLI override 列表，如 ['algo.lr=5e-4', 'device=cuda']。
                       通常由 main() 自动从 sys.argv 注入。
        silent: True 时不打印加载信息。

    返回:
        配置字典。

    用法:
        cfg = load_config()  # 自动查找 configs/phase1_default.yaml
        cfg = load_config("configs/my_config.yaml")
        cfg = load_config(cli_overrides=["algo.lr=5e-4", "env.num_envs=4"])
    """
    # 1. 确定配置文件路径
    if config_path is None:
        # 优先用环境变量，其次用 --config-file（如果已解析过）
        config_path = os.environ.get("CONFIG_FILE", _get_default_config_path())

    yaml_path = Path(config_path)
    if not yaml_path.exists():
        # 尝试相对于 cwd
        yaml_path_cwd = Path.cwd() / config_path
        if yaml_path_cwd.exists():
            yaml_path = yaml_path_cwd
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    # 2. 加载 YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    # 3. 应用 CLI overrides（从 sys.argv 自动解析）
    overrides = list(cli_overrides) if cli_overrides else []

    # 如果没有显式传入 cli_overrides，自动从 sys.argv 解析
    if cli_overrides is None:
        parsed, _ = argparse.ArgumentParser().parse_known_args()
        if hasattr(parsed, "config_overrides") and parsed.config_overrides:
            overrides.extend(parsed.config_overrides)
        if hasattr(parsed, "config_file") and parsed.config_file:
            # 重新加载指定的配置文件（优先级更高）
            with open(parsed.config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

    # 4. 应用所有 overrides
    for override in overrides:
        key_path, value = _parse_override(override)
        _set_nested(cfg, key_path, value)

    # 5. 打印加载结果（可关闭）
    if not silent:
        algo_name = cfg.get("algo", {}).get("algo_name", "unknown")
        env_name = cfg.get("env", {}).get("env_name", "unknown")
        print(f"[config] loaded from {yaml_path}  algo={algo_name}  env={env_name}")

    return cfg


def get_config_overrides_from_args(args: list[str]) -> list[str]:
    """从 args（如 sys.argv[1:]）提取 --config.KEY=VALUE overrides"""
    overrides = []
    for arg in args:
        if arg.startswith("--config="):
            overrides.append(arg[len("--config=") :])
        elif arg.startswith("--config ") and "=" in arg:
            # 处理 --config KEY=VALUE（空格分隔）
            overrides.append(arg.split("=", 1)[1])
    return overrides


# --- 便捷快捷函数 ---
def load_phase1_config(silent: bool = False) -> Dict[str, Any]:
    """加载 phase1 默认配置的快捷方式"""
    return load_config("configs/phase1_default.yaml", silent=silent)


if __name__ == "__main__":
    # 简单测试 / demo
    import pprint

    cfg = load_config()
    pprint.pprint(cfg)

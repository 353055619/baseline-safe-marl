"""
全局常量
"""
import torch

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 4

# 环境默认值
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_ENVS = 1

# 训练默认值
DEFAULT_EPISODES = 10
DEFAULT_BATCH_SIZE = 256

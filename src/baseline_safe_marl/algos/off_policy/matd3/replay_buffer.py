"""algos/matd3/replay_buffer.py - Multi-Agent Replay Buffer for MATD3"""
from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent off-policy algorithms (MATD3 / TD3).

    Stores transitions per-agent in flat arrays, indexed by step.
    Supports random mini-batch sampling for TD3-style updates.
    """

    def __init__(self, capacity: int = 100_000, batch_size: int = 256,
                 device: str = "cpu"):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self._ptr = 0
        self._size = 0
        self._obs_bufs: Dict[int, np.ndarray] = {}
        self._act_bufs: Dict[int, np.ndarray] = {}
        self._rew_bufs: Dict[int, np.ndarray] = {}
        self._next_obs_bufs: Dict[int, np.ndarray] = {}
        self._done_buf: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._num_agents = 0
        self._obs_dim = 0
        self._act_dim = 0
        self._initialized = False

    def _initialize_buffers(self, obs_dim: int, act_dim: int, num_agents: int):
        if self._initialized:
            return
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._num_agents = num_agents
        for i in range(num_agents):
            self._obs_bufs[i] = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self._act_bufs[i] = np.zeros((self.capacity, act_dim), dtype=np.float32)
            self._rew_bufs[i] = np.zeros(self.capacity, dtype=np.float32)
            self._next_obs_bufs[i] = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._initialized = True

    def add(self, obs_dict: Dict[int, np.ndarray],
            action_dict: Dict[int, np.ndarray],
            reward_dict: Dict[int, np.ndarray],
            next_obs_dict: Dict[int, np.ndarray],
            done_dict: Dict[int, float]):
        """Add one transition step for all agents."""
        if not self._initialized:
            first_aid = min(obs_dict.keys())
            obs_dim = np.array(obs_dict[first_aid], dtype=np.float32).shape[-1]
            act_dim = np.array(action_dict[first_aid], dtype=np.float32).shape[-1]
            self._initialize_buffers(obs_dim, act_dim, len(obs_dict))

        for aid in range(self._num_agents):
            self._obs_bufs[aid][self._ptr] = np.array(obs_dict[aid], dtype=np.float32).flatten()
            self._act_bufs[aid][self._ptr] = np.array(action_dict[aid], dtype=np.float32).flatten()
            self._rew_bufs[aid][self._ptr] = float(reward_dict[aid])
            self._next_obs_bufs[aid][self._ptr] = np.array(next_obs_dict[aid], dtype=np.float32).flatten()

        self._done_buf[self._ptr] = float(done_dict.get(0, 0.0))
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self) -> Dict[str, Any]:
        """Sample a random mini-batch."""
        indices = np.random.randint(0, self._size, size=self.batch_size)
        batch = {
            "obs":      {i: torch.as_tensor(self._obs_bufs[i][indices],      device=self.device, dtype=torch.float32)
                         for i in range(self._num_agents)},
            "action":   {i: torch.as_tensor(self._act_bufs[i][indices],       device=self.device, dtype=torch.float32)
                         for i in range(self._num_agents)},
            "reward":   {i: torch.as_tensor(self._rew_bufs[i][indices],       device=self.device, dtype=torch.float32)
                         for i in range(self._num_agents)},
            "next_obs": {i: torch.as_tensor(self._next_obs_bufs[i][indices], device=self.device, dtype=torch.float32)
                         for i in range(self._num_agents)},
            "done":     torch.as_tensor(self._done_buf[indices],             device=self.device, dtype=torch.float32),
            "indices":  indices,
        }
        return batch

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= self.batch_size

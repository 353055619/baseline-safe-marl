"""algos/matd3/trainer.py — MATD3 Trainer (with ReplayBuffer)"""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.optim as optim
import torch.nn.functional as F
from algos.base import BasePolicy, BaseTrainer
from algos.matd3.replay_buffer import MultiAgentReplayBuffer


class MATD3Trainer(BaseTrainer):
    """
    MATD3 trainer with ReplayBuffer and actual TD3 update logic.

    Key update steps per call:
      1. If buffer not ready: just count steps, return zeros
      2. Sample mini-batch from replay buffer
      3. Critic update (every step):
           - target_actions = clip(target_actor(next_obs) + noise, -1, 1)
           - y = r + gamma * min(Q1_target, Q2_target)(next_obs, target_actions)
           - critic_loss = MSE(Q1(obs,actions), y) + MSE(Q2(obs,actions), y)
           - step critic_opt
           - polyak soft-update target critics
      4. Actor update (every `delay` steps):
           - actor_loss = -mean(Q1_critic(obs, actor(obs)))
           - step actor_opt
           - hard-update target networks
    """

    def __init__(self, cfg: Dict[str, Any], policy: BasePolicy):
        super().__init__(cfg, policy)

        lr = cfg.get("algo", {}).get("lr", 1e-3)
        gamma = cfg.get("algo", {}).get("gamma", 0.99)
        self.gamma = gamma
        self.tau = policy.tau
        self._policy_delay = policy.delay
        self._update_counter = 0

        buffer_capacity = cfg.get("algo", {}).get("buffer_capacity", 100_000)
        batch_size = cfg.get("algo", {}).get("batch_size", 256)
        device = cfg.get("device", "cpu")
        self.buffer = MultiAgentReplayBuffer(
            capacity=buffer_capacity,
            batch_size=batch_size,
            device=device,
        )
        self._batch_size = batch_size

        # Target noise for action clipping (TD3 paper)
        self._target_noise = cfg.get("algo", {}).get("target_noise", 0.2)
        self._target_noise_clip = cfg.get("algo", {}).get("target_noise_clip", 0.5)

        # Optimizers
        self._actor_opt = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self._critic_opt = optim.Adam(
            list(self.policy.critic_q1.parameters())
            + list(self.policy.critic_q2.parameters()),
            lr=lr,
        )

    def add_transition(self, obs_dict, action_dict, reward_dict, next_obs_dict, done_dict):
        """Add one step of transition data to the replay buffer."""
        self.buffer.add(obs_dict, action_dict, reward_dict, next_obs_dict, done_dict)

    def train(self, num_steps: int) -> Dict[str, float]:
        self.total_steps += num_steps
        self._update_counter += num_steps

        if not self.buffer.is_ready:
            return {
                "critic_loss_q1": 0.0, "critic_loss_q2": 0.0,
                "actor_loss": 0.0, "q_value": 0.0,
                "policy_delay": float(self._policy_delay),
                "update_counter": float(self._update_counter),
                "buffer_size": float(len(self.buffer)),
            }

        # Sample mini-batch (one sample per step count — batch size may exceed)
        batch = self.buffer.sample()
        obs_t = batch["obs"]
        act_t = batch["action"]
        rew_t = batch["reward"]
        next_obs_t = batch["next_obs"]
        done_t = batch["done"]

        # ---- Critic update (every call) ----
        with torch.no_grad():
            # Target action: actor(next_obs) + clipped noise
            next_actions = {}
            for aid in obs_t.keys():
                na = self.policy.target_actor(next_obs_t[aid])
                noise = torch.randn_like(na) * self._target_noise
                noise = torch.clamp(noise, -self._target_noise_clip, self._target_noise_clip)
                next_actions[aid] = torch.clamp(na + noise, -1.0, 1.0)

            # Target Q values: r + gamma * min(Q1_target, Q2_target)(next_obs, target_action)
            q1_target_vals = []
            q2_target_vals = []
            for aid in obs_t.keys():
                q1t, q2t = self.policy.target_q1(next_obs_t[aid], next_actions[aid])
                q1_target_vals.append(q1t)
                q2_target_vals.append(q2t)

            # Average target across agents (shared reward signal)
            q1_target = torch.stack(q1_target_vals).mean(dim=0)
            q2_target = torch.stack(q2_target_vals).mean(dim=0)
            q_target_min = torch.min(q1_target, q2_target)

            rew_vals = torch.stack([rew_t[aid] for aid in rew_t.keys()]).mean(dim=0)
            y = rew_vals + self.gamma * q_target_min * (1.0 - done_t)

        # Current Q values — take mean across agents for shared-policy update
        q1_vals = []
        q2_vals = []
        for aid in obs_t.keys():
            q1, q2 = self.policy.critic_q1(obs_t[aid], act_t[aid])
            q1_vals.append(q1)
            q2_vals.append(q2)

        q1_mean = torch.stack(q1_vals).mean(dim=0)
        q2_mean = torch.stack(q2_vals).mean(dim=0)

        critic_loss_q1 = F.mse_loss(q1_mean, y)
        critic_loss_q2 = F.mse_loss(q2_mean, y)
        critic_loss = critic_loss_q1 + critic_loss_q2

        self._critic_opt.zero_grad()
        critic_loss.backward()
        self._critic_opt.step()

        # Soft update target critics (every step)
        self.policy.soft_update_target_networks()

        # ---- Actor update (every `delay` steps) ----
        actor_loss = torch.tensor(0.0)
        if self._update_counter % self._policy_delay == 0:
            actor_losses = []
            for aid in obs_t.keys():
                # Policy gradient: maximize Q1
                actions = self.policy.actor(obs_t[aid])
                q1_val = self.policy.critic_q1.q1(obs_t[aid], actions)
                actor_losses.append(-q1_val.mean())
            actor_loss = torch.stack(actor_losses).mean()

            self._actor_opt.zero_grad()
            actor_loss.backward()
            self._actor_opt.step()

            # Hard update target actor
            self.policy.hard_update_target_networks()

        return {
            "critic_loss_q1": float(critic_loss_q1.item()),
            "critic_loss_q2": float(critic_loss_q2.item()),
            "actor_loss": float(actor_loss.item()),
            "q_value": float(q1_mean.mean().item()),
            "policy_delay": float(self._policy_delay),
            "update_counter": float(self._update_counter),
            "buffer_size": float(len(self.buffer)),
        }

    def update_lagrangian(self) -> None:
        """No-op: MATD3 is unconstrained."""
        pass

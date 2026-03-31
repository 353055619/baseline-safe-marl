"""algos/facmac/trainer.py — FACMAC Trainer (with ReplayBuffer + full update)"""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.optim as optim
import torch.nn.functional as F
from algos.base import BasePolicy, BaseTrainer
from algos.matd3.replay_buffer import MultiAgentReplayBuffer


class FACMACTrainer(BaseTrainer):
    """
    FACMAC trainer with ReplayBuffer and full TD3-style update.

    Key update steps per call:
      1. If buffer not ready: count steps, return zeros
      2. Sample mini-batch
      3. Critic update (every step):
           - target_actions = clip(target_actor(next_obs) + noise, -1, 1)
           - Q_i_target = target_critic_i(next_obs, target_action_i)
           - Q_tot_target = target_mixing_net(Q_i_target, next_obs_state)
           - y = r + gamma * Q_tot_target * (1 - done)
           - critic_loss = MSE(Q_i(s, a_i), y) per agent + mixing_loss
           - step critic_opt
           - polyak soft-update target critics + mixing net
      4. Actor update (every `delay` steps):
           - actor_loss = -Q_tot_main(obs, actor(obs)) via deterministic PG
           - step actor_opt
           - hard-update all targets
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

        self._target_noise = cfg.get("algo", {}).get("target_noise", 0.2)
        self._target_noise_clip = cfg.get("algo", {}).get("target_noise_clip", 0.5)

        self._actor_opt = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self._critic_opt = optim.Adam(
            list(self.policy.mixing_net.parameters())
            + [p for c in self.policy.critics for p in c.parameters()],
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
                "critic_loss": 0.0, "mixing_loss": 0.0, "actor_loss": 0.0,
                "q_tot": 0.0, "policy_delay": float(self._policy_delay),
                "update_counter": float(self._update_counter),
                "buffer_size": float(len(self.buffer)),
            }

        batch = self.buffer.sample()
        obs_t = batch["obs"]
        act_t = batch["action"]
        rew_t = batch["reward"]
        next_obs_t = batch["next_obs"]
        done_t = batch["done"]

        # ---- Critic update ----
        n_agents = self.policy.n_agents

        with torch.no_grad():
            # Target actions with clipped noise
            next_actions = {}
            for aid in range(n_agents):
                na = self.policy.actor(next_obs_t[aid])
                noise = torch.randn_like(na) * self._target_noise
                noise = torch.clamp(noise, -self._target_noise_clip, self._target_noise_clip)
                next_actions[aid] = torch.clamp(na + noise, -1.0, 1.0)

            # Compute target Q_i for each agent
            q_target_vals = []
            for aid in range(n_agents):
                q_i_target = self.policy.target_critics[aid](
                    next_obs_t[aid], next_actions[aid]
                )
                q_target_vals.append(q_i_target)

            # Mixing: Q_tot = mixing_net(Q_i_vals, state) — use first agent's obs as state
            state_obs = torch.cat([next_obs_t[aid] for aid in range(n_agents)], dim=-1)
            q_tot_target = self.policy.target_mixing_net(
                torch.stack(q_target_vals, dim=-1),  # (B, n_agents)
                state_obs,
            ).squeeze(-1)

            # y = r + gamma * Q_tot_target * (1 - done)
            rew_vals = torch.stack([rew_t[aid] for aid in range(n_agents)]).mean(dim=0)
            y = rew_vals + self.gamma * q_tot_target * (1.0 - done_t)

        # Current Q_i values from main critics
        critic_losses = []
        for aid in range(n_agents):
            q_i = self.policy.critics[aid](obs_t[aid], act_t[aid]).squeeze(-1)
            critic_losses.append(F.mse_loss(q_i, y.detach()))

        mixing_out = self.policy.mixing_net(
            torch.stack([self.policy.critics[aid](obs_t[aid], act_t[aid])
                         for aid in range(n_agents)], dim=-1),
            torch.cat([obs_t[aid] for aid in range(n_agents)], dim=-1),
        )
        q_tot = mixing_out.squeeze(-1)
        mixing_loss = F.mse_loss(q_tot, y.detach())

        critic_loss = sum(critic_losses) / n_agents + mixing_loss

        self._critic_opt.zero_grad()
        critic_loss.backward()
        self._critic_opt.step()

        # Polyak soft-update targets
        self.policy.soft_update_targets()

        # ---- Actor update (every `delay` steps) ----
        actor_loss = torch.tensor(0.0)
        if self._update_counter % self._policy_delay == 0:
            actor_losses = []
            for aid in range(n_agents):
                actions = self.policy.actor(obs_t[aid])
                q_i = self.policy.critics[aid](obs_t[aid], actions).squeeze(-1)
                q_tot_local = self.policy.mixing_net(
                    torch.stack([self.policy.critics[i](obs_t[i],
                            (self.policy.actor(obs_t[i]) if i == aid else act_t[i]))
                                for i in range(n_agents)], dim=-1),
                    torch.cat([obs_t[aid] for aid in range(n_agents)], dim=-1),
                ).squeeze(-1)
                actor_losses.append(-q_tot_local.mean())

            actor_loss = torch.stack(actor_losses).mean()
            self._actor_opt.zero_grad()
            actor_loss.backward()
            self._actor_opt.step()

            self.policy.hard_update_targets()

        return {
            "critic_loss": float(critic_loss.item()),
            "mixing_loss": float(mixing_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_tot": float(q_tot.mean().item()),
            "policy_delay": float(self._policy_delay),
            "update_counter": float(self._update_counter),
            "buffer_size": float(len(self.buffer)),
        }

    def update_lagrangian(self) -> None:
        pass

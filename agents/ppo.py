from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional

from agents.networks import ActorCritic


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-length rollout buffer for on-policy PPO collection.

    Stores one scalar / array per step:
        states     (n_steps, *state_shape)
        actions    (n_steps,)
        rewards    (n_steps,)
        dones      (n_steps,)  — 1.0 if episode ended after this step
        log_probs  (n_steps,)  — log π(a|s) at collection time
        values     (n_steps,)  — V(s) at collection time
        advantages (n_steps,)  — filled by compute_returns_and_advantages()
        returns    (n_steps,)  — advantages + values
    """

    def __init__(self, n_steps: int, state_shape: tuple) -> None:
        self.n_steps = n_steps
        self.state_shape = state_shape
        self._ptr = 0

        self.states    = np.zeros((n_steps, *state_shape), dtype=np.float32)
        self.actions   = np.zeros(n_steps, dtype=np.int64)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        # valid action masks stored alongside states for loss computation
        self.valid_masks = np.zeros((n_steps, 7), dtype=np.float32)

    def reset(self) -> None:
        self._ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        valid_mask: np.ndarray,
    ) -> None:
        i = self._ptr
        self.states[i]     = state
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.dones[i]      = float(done)
        self.log_probs[i]  = log_prob
        self.values[i]     = value
        self.valid_masks[i] = valid_mask
        self._ptr += 1

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        """
        GAE-lambda advantage estimation (Schulman et al., 2015).

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            G_t = A_t + V(s_t)
        """
        gae = 0.0
        next_value = last_value

        for t in reversed(range(self.n_steps)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            next_value = self.values[t]

        self.returns = self.advantages + self.values

    def is_full(self) -> bool:
        return self._ptr >= self.n_steps


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPO:
    """
    Proximal Policy Optimization with:
      - Clipped surrogate objective
      - GAE-lambda advantage estimation
      - Entropy bonus
      - Invalid-action masking
    """

    def __init__(
        self,
        state_shape: tuple[int, ...] = (3, 6, 7),
        num_actions: int = 7,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 32,
    ) -> None:
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_eps    = clip_eps
        self.value_coef  = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps     = n_steps
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.num_actions = num_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network   = ActorCritic(num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = RolloutBuffer(n_steps, state_shape)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flip_board(state: np.ndarray) -> np.ndarray:
        """
        Swap channels 0 and 1 so the board is always seen from player 1's
        perspective regardless of whose turn it actually is.
        """
        flipped = state.copy()
        flipped[0], flipped[1] = state[1].copy(), state[0].copy()
        return flipped

    @torch.no_grad()
    def _policy_step(
        self, state: np.ndarray, valid_actions: list[int]
    ) -> tuple[int, float, float]:
        """
        Run one forward pass and sample an action.

        Returns:
            action    — sampled column index
            log_prob  — log π(action | state)
            value     — V(state)
        """
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.network(x)
        logits = logits.squeeze(0)

        mask = torch.full((self.num_actions,), -1e9, device=self.device)
        mask[valid_actions] = 0.0
        logits = logits + mask

        dist     = torch.distributions.Categorical(logits=logits)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)

        return int(action_t.item()), float(log_prob.item()), float(value.squeeze().item())

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env,
        opponent_fn: Optional[Callable[[np.ndarray, list[int]], int]] = None,
    ) -> None:
        """
        Collect exactly n_steps transitions for the agent (player 1).

        The environment alternates turns internally. After each opponent move
        the board is flipped so the agent always sees itself as player 1 in
        channel 0.

        opponent_fn: callable(state, valid_actions) -> action.
                     Defaults to uniform random if None.
        """
        if opponent_fn is None:
            opponent_fn = lambda s, v: int(np.random.choice(v))

        self.buffer.reset()

        raw_state = env.reset()
        # env starts with player 1; state is already from player 1's POV
        agent_state = raw_state

        while not self.buffer.is_full():
            valid = env.get_valid_actions()
            valid_mask = np.zeros(self.num_actions, dtype=np.float32)
            valid_mask[valid] = 1.0

            action, log_prob, value = self._policy_step(agent_state, valid)

            # Agent (player 1) acts
            raw_next, reward, done, info = env.step(action)

            if done:
                # Episode ended on agent's move (win, loss, draw, illegal)
                self.buffer.add(agent_state, action, reward, done, log_prob, value, valid_mask)
                raw_state   = env.reset()
                agent_state = raw_state
                continue

            # Opponent's turn
            opp_state  = self._flip_board(raw_next)   # opponent sees board from their POV
            opp_valid  = env.get_valid_actions()
            opp_action = opponent_fn(opp_state, opp_valid)
            raw_next2, opp_reward, done2, info2 = env.step(opp_action)

            if done2:
                # Opponent ended the episode — negate their reward for the agent
                agent_reward = -opp_reward
                agent_next   = self._flip_board(raw_next2)
                self.buffer.add(agent_state, action, agent_reward, True, log_prob, value, valid_mask)
                raw_state   = env.reset()
                agent_state = raw_state
            else:
                # Game continues; next agent state is opponent's next_state flipped
                agent_next = self._flip_board(raw_next2)
                self.buffer.add(agent_state, action, reward, False, log_prob, value, valid_mask)
                agent_state = agent_next

        # Bootstrap value for the last state
        with torch.no_grad():
            x = torch.from_numpy(agent_state).float().unsqueeze(0).to(self.device)
            _, last_value_t = self.network(x)
            last_value = float(last_value_t.squeeze().item())

        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates over the collected rollout.

        Returns a dict with averaged losses:
            policy_loss, value_loss, entropy, total_loss
        """
        states_np     = torch.from_numpy(self.buffer.states).float().to(self.device)
        actions_np    = torch.from_numpy(self.buffer.actions).long().to(self.device)
        old_log_probs = torch.from_numpy(self.buffer.log_probs).float().to(self.device)
        returns_np    = torch.from_numpy(self.buffer.returns).float().to(self.device)
        advantages_np = torch.from_numpy(self.buffer.advantages).float().to(self.device)
        masks_np      = torch.from_numpy(self.buffer.valid_masks).float().to(self.device)

        # Normalize advantages
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

        n = self.n_steps
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_loss_sum    = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                if len(idx) == 0:
                    continue

                sb = states_np[idx]
                ab = actions_np[idx]
                old_lp_b = old_log_probs[idx]
                ret_b    = returns_np[idx]
                adv_b    = advantages_np[idx]
                mask_b   = masks_np[idx]           # (B, 7)

                logits, values = self.network(sb)  # (B, 7), (B, 1)

                # Mask invalid actions
                logits = logits + (1.0 - mask_b) * -1e9

                dist     = torch.distributions.Categorical(logits=logits)
                new_lp_b = dist.log_prob(ab)
                entropy  = dist.entropy().mean()

                # Clipped surrogate policy loss
                ratio       = torch.exp(new_lp_b - old_lp_b)
                surr1       = ratio * adv_b
                surr2       = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                values = values.squeeze(1)
                value_loss = nn.functional.mse_loss(values, ret_b)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss  += float(value_loss.item())
                total_entropy     += float(entropy.item())
                total_loss_sum    += float(loss.item())
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss":  total_value_loss  / denom,
            "entropy":     total_entropy     / denom,
            "total_loss":  total_loss_sum    / denom,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "network":   self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.connect_k import ConnectK

    env   = ConnectK(k=4)
    agent = PPO(n_steps=128, batch_size=32, n_epochs=4)

    print(f"Device: {agent.device}")
    print("Collecting rollout against random opponent...")
    agent.collect_rollout(env, opponent_fn=None)

    print(f"  Rollout buffer full: {agent.buffer.is_full()}")
    print(f"  Advantages — mean: {agent.buffer.advantages.mean():.4f}  "
          f"std: {agent.buffer.advantages.std():.4f}")
    print(f"  Returns    — mean: {agent.buffer.returns.mean():.4f}  "
          f"std: {agent.buffer.returns.std():.4f}")

    print("\nRunning PPO update...")
    losses = agent.update()

    print("\nLoss breakdown:")
    for k, v in losses.items():
        print(f"  {k:14s}: {v:.6f}")

    assert "policy_loss" in losses
    assert "value_loss"  in losses
    assert "entropy"     in losses
    assert "total_loss"  in losses

    print("\nSmoke test passed.")

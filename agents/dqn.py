from __future__ import annotations

import copy
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import DuelingDQN
from utils.replay_buffer import PrioritizedReplayBuffer


class RainbowDQN:
    """
    Rainbow DQN agent combining:
      - Double DQN (online net selects, target net evaluates)
      - Dueling network architecture
      - Prioritized Experience Replay
      - N-step returns
    """

    def __init__(
        self,
        state_shape: tuple[int, ...] = (3, 6, 7),
        num_actions: int = 7,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        n_step: int = 3,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.online_net = DuelingDQN(num_actions).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=0.6,
            beta=beta_start,
        )

        # N-step buffer: deque of (state, action, reward, done)
        self._n_step_buf: deque = deque(maxlen=n_step)

        # Step counter (for target net sync and beta annealing)
        self._step = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray, valid_actions: list[int], epsilon: float
    ) -> int:
        """Epsilon-greedy action selection restricted to valid_actions."""
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))
        return self.online_net.get_action(state, valid_actions)

    # ------------------------------------------------------------------
    # N-step return helper
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Buffer a transition. When the n-step deque is full (or episode ends),
        compute the discounted n-step return and push to the replay buffer.
        """
        self._n_step_buf.append((state, action, reward, next_state, done))

        # Wait until we have n transitions, or flush on episode end
        if len(self._n_step_buf) < self.n_step and not done:
            return

        self._flush_n_step()

        # On episode end, keep flushing remaining transitions in the deque
        if done:
            while self._n_step_buf:
                self._flush_n_step()

    def _flush_n_step(self) -> None:
        """Pop the oldest transition and compute its n-step return."""
        if not self._n_step_buf:
            return

        s0, a0, _, _, _ = self._n_step_buf[0]

        # Accumulate discounted return from the available lookahead
        G = 0.0
        gamma_k = 1.0
        last_done = False
        last_next_state = self._n_step_buf[0][3]  # fallback

        for k, (_, _, r_k, ns_k, d_k) in enumerate(self._n_step_buf):
            G += gamma_k * r_k
            gamma_k *= self.gamma
            last_next_state = ns_k
            last_done = d_k
            if d_k:
                break  # episode ended; remaining entries belong to next episode

        self.buffer.push(s0, a0, G, last_next_state, last_done)
        self._n_step_buf.popleft()

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch, compute Double DQN loss with PER weights,
        update priorities, and sync target network if due.

        Returns loss as float, or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        self._step += 1

        # Anneal beta linearly from beta_start → 1.0 over beta_frames steps
        beta = min(1.0, self.beta_start + self._step * (1.0 - self.beta_start) / self.beta_frames)
        self.buffer.update_beta(beta)

        # Sample
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
            self.batch_size
        )

        states_t      = torch.from_numpy(states).float().to(self.device)
        actions_t     = torch.from_numpy(actions).long().to(self.device)
        rewards_t     = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t       = torch.from_numpy(dones).float().to(self.device)
        weights_t     = torch.from_numpy(weights).float().to(self.device)

        # --- Double DQN target ---
        with torch.no_grad():
            # Online net selects the best action in s'
            online_next_q = self.online_net(next_states_t)           # (B, A)
            # Mask invalid actions: channel 2 of next_state is the valid-move mask
            valid_mask = next_states_t[:, 2, 0, :]                   # (B, 7) — top-row free
            online_next_q = online_next_q + (1.0 - valid_mask) * -1e9
            best_actions = online_next_q.argmax(dim=1, keepdim=True)  # (B, 1)

            # Target net evaluates that action
            target_next_q = self.target_net(next_states_t)            # (B, A)
            target_next_q = target_next_q.gather(1, best_actions).squeeze(1)  # (B,)

            # n-step discounted target
            gamma_n = self.gamma ** self.n_step
            targets = rewards_t + gamma_n * target_next_q * (1.0 - dones_t)

        # --- Current Q-values ---
        q_values = self.online_net(states_t)                          # (B, A)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # --- Weighted Huber loss ---
        td_errors = targets - q_values
        loss_elementwise = nn.functional.huber_loss(q_values, targets, reduction="none")
        loss = (weights_t * loss_elementwise).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # --- Update PER priorities ---
        td_errors_np = td_errors.detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors_np)

        # --- Hard target network update ---
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._step,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step = checkpoint["step"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.connect_k import ConnectK

    env = ConnectK(k=4)
    agent = RainbowDQN(batch_size=64, buffer_capacity=10_000, target_update_freq=500)

    print(f"Device: {agent.device}")
    print("Pushing 200 random transitions...")

    state = env.reset()
    steps = 0
    episodes = 0

    while steps < 200:
        valid = env.get_valid_actions()
        action = agent.select_action(state, valid, epsilon=1.0)
        next_state, reward, done, _ = env.step(action)
        agent.push(state, action, reward, next_state, done)
        state = next_state
        steps += 1
        if done:
            state = env.reset()
            episodes += 1

    print(f"  {steps} steps, {episodes} episodes, buffer size: {len(agent.buffer)}")

    print("\nRunning 10 update steps...")
    losses = []
    for i in range(10):
        loss = agent.update()
        losses.append(loss)
        print(f"  update {i+1:2d}: loss = {loss:.6f}" if loss is not None else f"  update {i+1:2d}: buffer too small")

    assert any(l is not None for l in losses), "No update produced a loss"
    print("\nSmoke test passed.")

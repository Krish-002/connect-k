from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


NUM_ACTIONS = 7


# ---------------------------------------------------------------------------
# Shared convolutional backbone
# ---------------------------------------------------------------------------

class ConnectKEncoder(nn.Module):
    """
    Convolutional backbone shared by both DQN and PPO networks.

    Input:  (batch, 3, 6, 7)
    Output: (batch, 512)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 7, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Dueling DQN head
# ---------------------------------------------------------------------------

class DuelingDQNHead(nn.Module):
    """
    Dueling advantage/value streams (Wang et al., 2016).

    Input:  (batch, 512) encoder features
    Output: (batch, num_actions) Q-values
    """

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        v = self.value_stream(features)                    # (batch, 1)
        a = self.advantage_stream(features)                # (batch, num_actions)
        return v + a - a.mean(dim=1, keepdim=True)         # Q = V + A - mean(A)


# ---------------------------------------------------------------------------
# Actor-Critic head
# ---------------------------------------------------------------------------

class ActorCriticHead(nn.Module):
    """
    Separate actor (policy) and critic (value) heads for PPO.

    Input:  (batch, 512) encoder features
    Output: logits (batch, num_actions), value (batch, 1)
    """

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            # No softmax here — callers use log_softmax / Categorical
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(features), self.critic(features)


# ---------------------------------------------------------------------------
# Full networks
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """Rainbow DQN network: ConnectKEncoder + DuelingDQNHead."""

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.encoder = ConnectKEncoder()
        self.head = DuelingDQNHead(num_actions)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, num_actions)."""
        return self.head(self.encoder(x))

    @torch.no_grad()
    def get_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        """
        Select the greedy action restricted to valid_actions.

        Args:
            state:         (3, 6, 7) numpy array (single state, no batch dim)
            valid_actions: list of legal column indices

        Returns:
            action as int
        """
        x = torch.from_numpy(state).float().unsqueeze(0)   # (1, 3, 6, 7)
        q = self.forward(x).squeeze(0)                     # (num_actions,)

        mask = torch.full((self.num_actions,), -1e9)
        mask[valid_actions] = 0.0
        q = q + mask

        return int(q.argmax().item())


class ActorCritic(nn.Module):
    """PPO network: ConnectKEncoder + ActorCriticHead."""

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.encoder = ConnectKEncoder()
        self.head = ActorCriticHead(num_actions)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value) with shapes (batch, num_actions) and (batch, 1)."""
        return self.head(self.encoder(x))

    @torch.no_grad()
    def get_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        """
        Sample an action from the masked policy.

        Invalid actions are set to -1e9 before softmax so they get ~0 probability.

        Args:
            state:         (3, 6, 7) numpy array (single state, no batch dim)
            valid_actions: list of legal column indices

        Returns:
            action as int
        """
        x = torch.from_numpy(state).float().unsqueeze(0)   # (1, 3, 6, 7)
        logits, _ = self.forward(x)
        logits = logits.squeeze(0)                          # (num_actions,)

        mask = torch.full((self.num_actions,), -1e9)
        mask[valid_actions] = 0.0
        logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        return int(action)


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH = 4
    x = torch.randn(BATCH, 3, 6, 7)

    # Encoder
    enc = ConnectKEncoder()
    features = enc(x)
    assert features.shape == (BATCH, 512), f"Encoder: {features.shape}"
    print(f"ConnectKEncoder output:      {tuple(features.shape)}")

    # DuelingDQN
    dqn = DuelingDQN()
    q = dqn(x)
    assert q.shape == (BATCH, NUM_ACTIONS), f"DQN Q-values: {q.shape}"
    print(f"DuelingDQN Q-values:         {tuple(q.shape)}")

    # ActorCritic
    ac = ActorCritic()
    logits, value = ac(x)
    assert logits.shape == (BATCH, NUM_ACTIONS), f"AC logits: {logits.shape}"
    assert value.shape == (BATCH, 1), f"AC value: {value.shape}"
    print(f"ActorCritic logits:          {tuple(logits.shape)}")
    print(f"ActorCritic value:           {tuple(value.shape)}")

    # get_action (single state)
    state = x[0].numpy()
    valid = [0, 2, 4, 6]

    dqn_action = dqn.get_action(state, valid)
    assert dqn_action in valid, f"DQN chose invalid action {dqn_action}"
    print(f"\nDuelingDQN.get_action:       {dqn_action}  (valid={valid})")

    ac_action = ac.get_action(state, valid)
    assert ac_action in valid, f"AC chose invalid action {ac_action}"
    print(f"ActorCritic.get_action:      {ac_action}  (valid={valid})")

    total_dqn = sum(p.numel() for p in dqn.parameters())
    total_ac  = sum(p.numel() for p in ac.parameters())
    print(f"\nDuelingDQN parameters:       {total_dqn:,}")
    print(f"ActorCritic parameters:      {total_ac:,}")

    print("\nAll shape checks passed.")

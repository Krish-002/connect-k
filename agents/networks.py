from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


NUM_ACTIONS = 7


class ConnectKEncoder(nn.Module):
    # shared conv backbone, input (B, 3, 6, 7) -> (B, 512)

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


class DuelingDQNHead(nn.Module):
    # Q = V + A - mean(A), Wang et al. 2016

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
        v = self.value_stream(features)
        a = self.advantage_stream(features)
        return v + a - a.mean(dim=1, keepdim=True)


class ActorCriticHead(nn.Module):

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            # no softmax — callers use Categorical(logits=...)
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(features), self.critic(features)


class DuelingDQN(nn.Module):

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.encoder = ConnectKEncoder()
        self.head = DuelingDQNHead(num_actions)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def get_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        x = torch.from_numpy(state).float().unsqueeze(0)
        q = self.forward(x).squeeze(0)

        mask = torch.full((self.num_actions,), -1e9)
        mask[valid_actions] = 0.0
        q = q + mask

        return int(q.argmax().item())


class ActorCritic(nn.Module):

    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.encoder = ConnectKEncoder()
        self.head = ActorCriticHead(num_actions)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def get_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        x = torch.from_numpy(state).float().unsqueeze(0)
        logits, _ = self.forward(x)
        logits = logits.squeeze(0)

        mask = torch.full((self.num_actions,), -1e9)
        mask[valid_actions] = 0.0
        logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


if __name__ == "__main__":
    BATCH = 4
    x = torch.randn(BATCH, 3, 6, 7)

    enc = ConnectKEncoder()
    features = enc(x)
    assert features.shape == (BATCH, 512)
    print(f"encoder: {tuple(features.shape)}")

    dqn = DuelingDQN()
    q = dqn(x)
    assert q.shape == (BATCH, NUM_ACTIONS)
    print(f"DuelingDQN Q-values: {tuple(q.shape)}")

    ac = ActorCritic()
    logits, value = ac(x)
    assert logits.shape == (BATCH, NUM_ACTIONS)
    assert value.shape == (BATCH, 1)
    print(f"ActorCritic logits: {tuple(logits.shape)}  value: {tuple(value.shape)}")

    state = x[0].numpy()
    valid = [0, 2, 4, 6]

    dqn_action = dqn.get_action(state, valid)
    ac_action = ac.get_action(state, valid)
    assert dqn_action in valid
    assert ac_action in valid
    print(f"DQN action: {dqn_action}  AC action: {ac_action}  (valid={valid})")

    print(f"DQN params: {sum(p.numel() for p in dqn.parameters()):,}")
    print(f"AC params:  {sum(p.numel() for p in ac.parameters()):,}")

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.ppo import PPO
from env.connect_k import ConnectK

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS    = 500_000
N_STEPS            = 128       # rollout steps collected per update
EVAL_FREQ          = 10_000    # timesteps between evaluations
SAVE_FREQ          = 50_000    # timesteps between checkpoints
K                  = 4
SELF_PLAY_START    = 50_000    # timestep to switch from random to self-play
FROZEN_UPDATE_FREQ = 25_000    # re-freeze opponent every N timesteps after self-play starts

CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flip_board(state: np.ndarray) -> np.ndarray:
    """Swap channels 0/1 so the active player always sees itself in channel 0."""
    flipped = state.copy()
    flipped[0], flipped[1] = state[1].copy(), state[0].copy()
    return flipped


def _make_random_opponent():
    def _random(state: np.ndarray, valid_actions: list[int]) -> int:
        return int(np.random.choice(valid_actions))
    return _random


def _make_frozen_opponent(agent: PPO, epsilon: float = 0.05):
    """
    Freeze a copy of the agent's network.  Acts greedily (argmax on logits)
    with a small exploration epsilon so the agent doesn't overfit to one style.
    """
    frozen_net = copy.deepcopy(agent.network)
    frozen_net.eval()
    num_actions = agent.num_actions
    device = agent.device

    @torch.no_grad()
    def _act(state: np.ndarray, valid_actions: list[int]) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits, _ = frozen_net(x)
        logits = logits.squeeze(0)
        mask = torch.full((num_actions,), -1e9, device=device)
        mask[valid_actions] = 0.0
        logits = logits + mask
        return int(logits.argmax().item())

    return _act


def play_episode_greedy(
    agent: PPO,
    env: ConnectK,
    opponent_fn,
) -> int:
    """
    Play one episode with the agent acting greedily (argmax on masked logits).

    Returns outcome: +1 win, -1 loss, 0 draw.
    """
    raw_state = env.reset()
    agent_state = raw_state
    device = agent.device

    while True:
        valid = env.get_valid_actions()

        # Greedy action
        with torch.no_grad():
            x = torch.from_numpy(agent_state).float().unsqueeze(0).to(device)
            logits, _ = agent.network(x)
            logits = logits.squeeze(0)
            mask = torch.full((agent.num_actions,), -1e9, device=device)
            mask[valid] = 0.0
            logits = logits + mask
            action = int(logits.argmax().item())

        raw_next, reward, done, info = env.step(action)

        if done:
            if info["winner"] == 1:
                return 1
            elif info["winner"] == -1:
                return -1
            return 0

        # Opponent moves
        opp_state  = _flip_board(raw_next)
        opp_valid  = env.get_valid_actions()
        opp_action = opponent_fn(opp_state, opp_valid)
        raw_next2, opp_reward, done2, info2 = env.step(opp_action)

        if done2:
            if info2["winner"] == -1:
                return -1   # opponent (player -1) won
            elif info2["winner"] == 1:
                return 1
            return 0

        agent_state = _flip_board(raw_next2)


def eval_agent(
    agent: PPO,
    env: ConnectK,
    opponent_fn,
    n_games: int = 100,
) -> dict[str, float]:
    """
    Evaluate the agent greedily over n_games episodes.

    Returns dict with win_rate, loss_rate, draw_rate.
    """
    wins = losses = draws = 0
    for _ in range(n_games):
        outcome = play_episode_greedy(agent, env, opponent_fn)
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            draws += 1
    return {
        "win_rate":  wins   / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws  / n_games,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(k: int = K) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env   = ConnectK(k=k)
    agent = PPO(n_steps=N_STEPS)

    opponent_fn          = _make_random_opponent()
    using_self_play      = False
    last_freeze_timestep = 0

    total_timesteps = 0
    update_count    = 0

    # Loss accumulators between eval windows
    acc: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []
    }
    log: list[dict] = []

    # Determine the next eval and save thresholds
    next_eval = EVAL_FREQ
    next_save = SAVE_FREQ

    print(f"Training PPO on Connect-{k}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS}  |  Self-play from step {SELF_PLAY_START}")
    print("-" * 72)

    while total_timesteps < TOTAL_TIMESTEPS:

        # ---- Self-play switching / re-freezing ----
        if not using_self_play and total_timesteps >= SELF_PLAY_START:
            opponent_fn     = _make_frozen_opponent(agent)
            using_self_play = True
            last_freeze_timestep = total_timesteps
            print(f"  [step {total_timesteps}] Switched to self-play opponent.")

        if using_self_play and (total_timesteps - last_freeze_timestep) >= FROZEN_UPDATE_FREQ:
            opponent_fn          = _make_frozen_opponent(agent)
            last_freeze_timestep = total_timesteps

        # ---- Collect rollout then update ----
        agent.collect_rollout(env, opponent_fn)
        losses = agent.update()
        update_count    += 1
        total_timesteps += N_STEPS

        for key in acc:
            acc[key].append(losses[key])

        # ---- Evaluation & logging ----
        if total_timesteps >= next_eval:
            eval_results = eval_agent(agent, env, opponent_fn, n_games=100)

            avg = {k_: float(np.mean(v)) if v else float("nan") for k_, v in acc.items()}

            print(
                f"Step {total_timesteps:7d} | "
                f"Win: {eval_results['win_rate']*100:5.1f}% "
                f"Loss: {eval_results['loss_rate']*100:5.1f}% "
                f"Draw: {eval_results['draw_rate']*100:5.1f}% | "
                f"PLoss: {avg['policy_loss']:7.4f} "
                f"VLoss: {avg['value_loss']:7.4f} "
                f"Ent: {avg['entropy']:6.4f}"
            )

            log_entry = {
                "timestep":    total_timesteps,
                "update":      update_count,
                "policy_loss": round(avg["policy_loss"], 6),
                "value_loss":  round(avg["value_loss"],  6),
                "entropy":     round(avg["entropy"],     6),
                "total_loss":  round(avg["total_loss"],  6),
                **{k_: round(v, 4) for k_, v in eval_results.items()},
            }
            log.append(log_entry)

            with open(os.path.join(LOG_DIR, "ppo_log.json"), "w") as f:
                json.dump(log, f, indent=2)

            # Reset accumulators and advance threshold
            for key in acc:
                acc[key].clear()
            next_eval += EVAL_FREQ

        # ---- Checkpoint ----
        if total_timesteps >= next_save:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ppo_step{total_timesteps:07d}.pt")
            agent.save(ckpt_path)
            print(f"  [step {total_timesteps}] Checkpoint saved → {ckpt_path}")
            next_save += SAVE_FREQ

    print("-" * 72)
    print("Training complete.")

    final_path = os.path.join(CHECKPOINT_DIR, "ppo_final.pt")
    agent.save(final_path)
    print(f"Final model saved → {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Connect-K")
    parser.add_argument("--k", type=int, default=K, help="Win condition (default: 4)")
    args = parser.parse_args()

    train(k=args.k)

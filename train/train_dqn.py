from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

# Allow running as a script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.dqn import RainbowDQN
from env.connect_k import ConnectK

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPISODES         = 5000
EPSILON_START    = 1.0
EPSILON_END      = 0.05
EPSILON_DECAY    = 0.9995
EVAL_FREQ        = 250    # episodes between evaluation / logging
SAVE_FREQ        = 500    # episodes between checkpoint saves
K                = 4      # connect-k win condition
SELF_PLAY_START  = 500    # episode index to switch from random to self-play
SELF_PLAY_UPDATE = 250    # re-freeze opponent every N episodes after self-play starts

CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flip_board(state: np.ndarray) -> np.ndarray:
    """Swap channels 0 and 1 so the viewing player sees itself as channel 0."""
    flipped = state.copy()
    flipped[0], flipped[1] = state[1].copy(), state[0].copy()
    return flipped


def _make_random_opponent():
    def _random(state: np.ndarray, valid_actions: list[int]) -> int:
        return int(np.random.choice(valid_actions))
    return _random


def _make_frozen_opponent(agent: RainbowDQN, epsilon: float = 0.05):
    """Return a callable that acts greedily (with small epsilon) using a deep copy."""
    frozen_net = copy.deepcopy(agent.online_net)
    frozen_net.eval()

    def _act(state: np.ndarray, valid_actions: list[int]) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))
        return frozen_net.get_action(state, valid_actions)

    return _act


def play_episode(
    env: ConnectK,
    agent: RainbowDQN,
    opponent_fn,
    epsilon: float,
) -> tuple[float, int]:
    """
    Play one full episode.

    Agent is always player 1.  After each opponent move the board is flipped
    so the agent always observes itself in channel 0.

    Returns:
        total_reward  — sum of rewards received by the agent
        outcome       — +1 win, -1 loss, 0 draw
    """
    raw_state = env.reset()
    agent_state = raw_state
    total_reward = 0.0
    outcome = 0

    while True:
        valid = env.get_valid_actions()
        action = agent.select_action(agent_state, valid, epsilon)

        raw_next, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            agent.push(agent_state, action, reward, _flip_board(raw_next), done)
            if info["winner"] == 1:
                outcome = 1
            elif info["winner"] == -1:
                outcome = -1
            else:
                outcome = 0
            break

        # Opponent's turn
        opp_state  = _flip_board(raw_next)
        opp_valid  = env.get_valid_actions()
        opp_action = opponent_fn(opp_state, opp_valid)

        raw_next2, opp_reward, done2, info2 = env.step(opp_action)

        if done2:
            agent_reward = -opp_reward
            total_reward += agent_reward
            agent_next = _flip_board(raw_next2)
            agent.push(agent_state, action, agent_reward, agent_next, True)
            # Opponent won → agent lost; opponent drew → draw
            if info2["winner"] == -1:
                outcome = -1
            elif info2["winner"] == 1:
                # Shouldn't happen (opponent is player -1), but be safe
                outcome = 1
            else:
                outcome = 0
            break

        agent_next = _flip_board(raw_next2)
        agent.push(agent_state, action, reward, agent_next, False)
        agent_state = agent_next

    return total_reward, outcome


def eval_agent(
    agent: RainbowDQN,
    env: ConnectK,
    opponent_fn,
    n_games: int = 100,
) -> dict[str, float]:
    """
    Evaluate the agent greedily (epsilon=0) over n_games episodes.

    Returns dict with win_rate, loss_rate, draw_rate.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        _, outcome = play_episode(env, agent, opponent_fn, epsilon=0.0)
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

    env  = ConnectK(k=k)
    agent = RainbowDQN()

    opponent_fn   = _make_random_opponent()
    using_self_play = False
    last_self_play_update = 0

    epsilon = EPSILON_START
    log: list[dict] = []

    # Accumulators reset every EVAL_FREQ episodes
    recent_losses: list[float] = []
    recent_wins = recent_losses_count = recent_draws = 0

    print(f"Training Rainbow DQN on Connect-{k}")
    print(f"  Episodes: {EPISODES}  |  Self-play from ep {SELF_PLAY_START}")
    print("-" * 65)

    for ep in range(1, EPISODES + 1):

        # ---- Self-play switching / re-freezing ----
        if ep == SELF_PLAY_START and not using_self_play:
            opponent_fn = _make_frozen_opponent(agent)
            using_self_play = True
            last_self_play_update = ep
            print(f"  [ep {ep}] Switched to self-play opponent.")

        if using_self_play and (ep - last_self_play_update) >= SELF_PLAY_UPDATE:
            opponent_fn = _make_frozen_opponent(agent)
            last_self_play_update = ep

        # ---- Play one episode ----
        _, outcome = play_episode(env, agent, opponent_fn, epsilon)

        if outcome == 1:
            recent_wins += 1
        elif outcome == -1:
            recent_losses_count += 1
        else:
            recent_draws += 1

        # ---- Learning updates (one per step is handled inside push/update) ----
        loss = agent.update()
        if loss is not None:
            recent_losses.append(loss)

        # ---- Epsilon decay ----
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # ---- Evaluation & logging ----
        if ep % EVAL_FREQ == 0:
            eval_results = eval_agent(agent, env, opponent_fn, n_games=100)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            total_recent = recent_wins + recent_losses_count + recent_draws
            train_win  = recent_wins          / max(total_recent, 1)
            train_loss = recent_losses_count  / max(total_recent, 1)
            train_draw = recent_draws         / max(total_recent, 1)

            print(
                f"Episode {ep:5d} | "
                f"Win: {eval_results['win_rate']*100:5.1f}% "
                f"Loss: {eval_results['loss_rate']*100:5.1f}% "
                f"Draw: {eval_results['draw_rate']*100:5.1f}% | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Eps: {epsilon:.3f}"
            )

            log_entry = {
                "episode":    ep,
                "epsilon":    round(epsilon, 4),
                "avg_loss":   round(avg_loss, 6),
                "train_win":  round(train_win,  4),
                "train_loss": round(train_loss, 4),
                "train_draw": round(train_draw, 4),
                **{k_: round(v, 4) for k_, v in eval_results.items()},
            }
            log.append(log_entry)

            with open(os.path.join(LOG_DIR, "dqn_log.json"), "w") as f:
                json.dump(log, f, indent=2)

            # Reset accumulators
            recent_losses = []
            recent_wins = recent_losses_count = recent_draws = 0

        # ---- Checkpoint ----
        if ep % SAVE_FREQ == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{ep:05d}.pt")
            agent.save(ckpt_path)
            print(f"  [ep {ep}] Checkpoint saved → {ckpt_path}")

    print("-" * 65)
    print("Training complete.")

    # Final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, "dqn_final.pt")
    agent.save(final_path)
    print(f"Final model saved → {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rainbow DQN on Connect-K")
    parser.add_argument("--k", type=int, default=K, help="Win condition (default: 4)")
    args = parser.parse_args()

    train(k=args.k)

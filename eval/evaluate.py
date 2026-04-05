from __future__ import annotations

import json
import os
import sys
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.connect_k import ConnectK
from agents.dqn import RainbowDQN
from agents.ppo import PPO


# ---------------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------------

def _flip_board(state: np.ndarray) -> np.ndarray:
    """Swap channels 0/1 so the viewer always sees itself in channel 0."""
    flipped = state.copy()
    flipped[0], flipped[1] = state[1].copy(), state[0].copy()
    return flipped


# ---------------------------------------------------------------------------
# Greedy action wrappers
# ---------------------------------------------------------------------------

def _dqn_greedy(agent: RainbowDQN) -> Callable[[np.ndarray, list[int]], int]:
    """Return a callable that acts greedily with a DQN agent."""
    def _act(state: np.ndarray, valid_actions: list[int]) -> int:
        return agent.online_net.get_action(state, valid_actions)
    return _act


def _ppo_greedy(agent: PPO) -> Callable[[np.ndarray, list[int]], int]:
    """Return a callable that acts greedily (argmax logits) with a PPO agent."""
    num_actions = agent.num_actions
    device = agent.device

    @torch.no_grad()
    def _act(state: np.ndarray, valid_actions: list[int]) -> int:
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits, _ = agent.network(x)
        logits = logits.squeeze(0)
        mask = torch.full((num_actions,), -1e9, device=device)
        mask[valid_actions] = 0.0
        logits = logits + mask
        return int(logits.argmax().item())

    return _act


def _agent_greedy_fn(agent) -> Callable[[np.ndarray, list[int]], int]:
    """Dispatch to the right greedy wrapper based on agent type."""
    if isinstance(agent, RainbowDQN):
        return _dqn_greedy(agent)
    if isinstance(agent, PPO):
        return _ppo_greedy(agent)
    raise TypeError(f"Unknown agent type: {type(agent)}")


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def _minimax(
    env: ConnectK,
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    k: int,
) -> float:
    """
    Depth-limited minimax on a raw board array (6×7, values +1/-1/0).
    +1 means the *minimax player* (player -1 in env terms) wins.

    We work directly on a board copy to avoid resetting the live env.
    """
    # Check terminal: did the last move win? We check all four directions.
    # Rather than reimplementing win detection, we look for k-in-a-row.
    def _has_won(b: np.ndarray, player: int) -> bool:
        rows, cols = b.shape
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(rows):
            for c in range(cols):
                if b[r, c] != player:
                    continue
                for dr, dc in dirs:
                    count = 0
                    nr, nc = r, c
                    while 0 <= nr < rows and 0 <= nc < cols and b[nr, nc] == player:
                        count += 1
                        nr += dr
                        nc += dc
                    if count >= k:
                        return True
        return False

    def _valid_cols(b: np.ndarray) -> list[int]:
        return [c for c in range(b.shape[1]) if b[0, c] == 0]

    def _drop(b: np.ndarray, col: int, player: int) -> np.ndarray:
        nb = b.copy()
        for r in range(b.shape[0] - 1, -1, -1):
            if nb[r, col] == 0:
                nb[r, col] = player
                break
        return nb

    # The minimax agent is player -1 (opponent); agent is player 1 (minimizer).
    # maximizing=True  → it's the minimax agent's turn (player -1)
    # maximizing=False → it's the real agent's turn  (player  1)
    minimax_player  = -1
    minimizer_player = 1

    valid = _valid_cols(board)

    if _has_won(board, minimax_player):
        return 1.0
    if _has_won(board, minimizer_player):
        return -1.0
    if not valid or depth == 0:
        return 0.0

    if maximizing:
        best = -float("inf")
        for col in valid:
            nb = _drop(board, col, minimax_player)
            val = _minimax(env, nb, depth - 1, alpha, beta, False, k)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float("inf")
        for col in valid:
            nb = _drop(board, col, minimizer_player)
            val = _minimax(env, nb, depth - 1, alpha, beta, True, k)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def _minimax_action(env: ConnectK, board: np.ndarray, depth: int, k: int) -> int:
    """Pick the column that maximises the minimax value for player -1."""
    def _valid_cols(b: np.ndarray) -> list[int]:
        return [c for c in range(b.shape[1]) if b[0, c] == 0]

    def _drop(b: np.ndarray, col: int, player: int) -> np.ndarray:
        nb = b.copy()
        for r in range(b.shape[0] - 1, -1, -1):
            if nb[r, col] == 0:
                nb[r, col] = player
                break
        return nb

    valid = _valid_cols(board)
    best_val, best_col = -float("inf"), valid[0]
    for col in valid:
        nb = _drop(board, col, -1)
        val = _minimax(env, nb, depth - 1, -float("inf"), float("inf"), False, k)
        if val > best_val:
            best_val = val
            best_col = col
    return best_col


# ---------------------------------------------------------------------------
# Head-to-head evaluation
# ---------------------------------------------------------------------------

def head_to_head(
    agent1,
    agent2,
    env: ConnectK,
    n_games: int = 200,
) -> dict[str, float | int]:
    """
    Play n_games games between agent1 and agent2.

    For the first n_games//2 games agent1 goes first (player 1).
    For the remaining games agent2 goes first (player 1).
    Both agents play greedily.

    Returns dict: agent1_wins, agent2_wins, draws, agent1_win_rate.
    """
    fn1 = _agent_greedy_fn(agent1)
    fn2 = _agent_greedy_fn(agent2)

    agent1_wins = agent2_wins = draws = 0
    half = n_games // 2

    for game_idx in range(n_games):
        # Alternate who goes first each half
        if game_idx < half:
            first_fn, second_fn = fn1, fn2
            first_is_agent1 = True
        else:
            first_fn, second_fn = fn2, fn1
            first_is_agent1 = False

        raw_state = env.reset()
        first_state = raw_state  # first player's perspective

        winner_is_first = None  # True/False/None(draw)

        while True:
            valid = env.get_valid_actions()
            action = first_fn(first_state, valid)
            raw_next, reward, done, info = env.step(action)

            if done:
                if info["winner"] == 1:
                    winner_is_first = True
                elif info["winner"] == -1:
                    winner_is_first = False
                else:
                    winner_is_first = None
                break

            second_state = _flip_board(raw_next)
            valid2 = env.get_valid_actions()
            action2 = second_fn(second_state, valid2)
            raw_next2, _, done2, info2 = env.step(action2)

            if done2:
                if info2["winner"] == -1:
                    # Second player (who is player -1) won
                    winner_is_first = False
                elif info2["winner"] == 1:
                    winner_is_first = True
                else:
                    winner_is_first = None
                break

            first_state = _flip_board(raw_next2)

        # Map winner back to agent1/agent2
        if winner_is_first is None:
            draws += 1
        elif winner_is_first == first_is_agent1:
            agent1_wins += 1
        else:
            agent2_wins += 1

    return {
        "agent1_wins":    agent1_wins,
        "agent2_wins":    agent2_wins,
        "draws":          draws,
        "agent1_win_rate": agent1_wins / n_games,
    }


# ---------------------------------------------------------------------------
# Agent vs Minimax
# ---------------------------------------------------------------------------

def vs_minimax(
    agent,
    env: ConnectK,
    depth: int = 3,
    n_games: int = 50,
) -> dict[str, float]:
    """
    Play n_games where the agent is player 1 and minimax is player -1.
    Agent plays greedily; minimax uses alpha-beta pruning to depth `depth`.

    Returns dict: win_rate, loss_rate, draw_rate.
    """
    agent_fn = _agent_greedy_fn(agent)
    k = env.k
    wins = losses = draws = 0

    for _ in range(n_games):
        raw_state = env.reset()
        agent_state = raw_state

        while True:
            valid = env.get_valid_actions()
            action = agent_fn(agent_state, valid)
            raw_next, reward, done, info = env.step(action)

            if done:
                if info["winner"] == 1:
                    wins += 1
                elif info["winner"] == -1:
                    losses += 1
                else:
                    draws += 1
                break

            # Minimax picks a column using the raw board (player -1 maximizes)
            mm_action = _minimax_action(env, env._board.copy(), depth, k)
            raw_next2, _, done2, info2 = env.step(mm_action)

            if done2:
                if info2["winner"] == -1:
                    losses += 1
                elif info2["winner"] == 1:
                    wins += 1
                else:
                    draws += 1
                break

            agent_state = _flip_board(raw_next2)

    return {
        "win_rate":  wins   / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws  / n_games,
    }


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def _random_fn(state: np.ndarray, valid_actions: list[int]) -> int:
    return int(np.random.choice(valid_actions))


def run_full_eval(
    dqn_path: str,
    ppo_path: str,
    k: int = 4,
) -> dict:
    """
    Load both agents, run all matchups, print a results table, and save JSON.
    """
    env = ConnectK(k=k)

    # Load agents
    dqn = RainbowDQN()
    dqn.load(dqn_path)
    dqn.online_net.eval()

    ppo = PPO()
    ppo.load(ppo_path)
    ppo.network.eval()

    print(f"\n{'='*58}")
    print(f"  Evaluation — Connect-{k}")
    print(f"{'='*58}")

    # DQN vs random
    print("DQN vs Random (100 games)...", end=" ", flush=True)
    dqn_vs_random = vs_minimax.__wrapped__ if hasattr(vs_minimax, "__wrapped__") else None
    # reuse head_to_head logic via a random agent wrapper
    _rand_wins = _rand_losses = _rand_draws = 0
    dqn_fn = _agent_greedy_fn(dqn)
    for _ in range(100):
        raw = env.reset(); s = raw
        outcome = 0
        while True:
            v = env.get_valid_actions()
            a = dqn_fn(s, v)
            nxt, r, done, info = env.step(a)
            if done:
                outcome = 1 if info["winner"] == 1 else (-1 if info["winner"] == -1 else 0)
                break
            opp_a = _random_fn(_flip_board(nxt), env.get_valid_actions())
            nxt2, _, done2, info2 = env.step(opp_a)
            if done2:
                outcome = -1 if info2["winner"] == -1 else (1 if info2["winner"] == 1 else 0)
                break
            s = _flip_board(nxt2)
        if outcome == 1: _rand_wins += 1
        elif outcome == -1: _rand_losses += 1
        else: _rand_draws += 1
    dqn_vs_random = {"win_rate": _rand_wins/100, "loss_rate": _rand_losses/100, "draw_rate": _rand_draws/100}
    print(f"W={dqn_vs_random['win_rate']:.0%} L={dqn_vs_random['loss_rate']:.0%} D={dqn_vs_random['draw_rate']:.0%}")

    # PPO vs random
    print("PPO vs Random (100 games)...", end=" ", flush=True)
    _rand_wins = _rand_losses = _rand_draws = 0
    ppo_fn = _agent_greedy_fn(ppo)
    for _ in range(100):
        raw = env.reset(); s = raw
        outcome = 0
        while True:
            v = env.get_valid_actions()
            a = ppo_fn(s, v)
            nxt, r, done, info = env.step(a)
            if done:
                outcome = 1 if info["winner"] == 1 else (-1 if info["winner"] == -1 else 0)
                break
            opp_a = _random_fn(_flip_board(nxt), env.get_valid_actions())
            nxt2, _, done2, info2 = env.step(opp_a)
            if done2:
                outcome = -1 if info2["winner"] == -1 else (1 if info2["winner"] == 1 else 0)
                break
            s = _flip_board(nxt2)
        if outcome == 1: _rand_wins += 1
        elif outcome == -1: _rand_losses += 1
        else: _rand_draws += 1
    ppo_vs_random = {"win_rate": _rand_wins/100, "loss_rate": _rand_losses/100, "draw_rate": _rand_draws/100}
    print(f"W={ppo_vs_random['win_rate']:.0%} L={ppo_vs_random['loss_rate']:.0%} D={ppo_vs_random['draw_rate']:.0%}")

    # DQN vs minimax
    print("DQN vs Minimax depth=3 (50 games)...", end=" ", flush=True)
    dqn_vs_mm = vs_minimax(dqn, env, depth=3, n_games=50)
    print(f"W={dqn_vs_mm['win_rate']:.0%} L={dqn_vs_mm['loss_rate']:.0%} D={dqn_vs_mm['draw_rate']:.0%}")

    # PPO vs minimax
    print("PPO vs Minimax depth=3 (50 games)...", end=" ", flush=True)
    ppo_vs_mm = vs_minimax(ppo, env, depth=3, n_games=50)
    print(f"W={ppo_vs_mm['win_rate']:.0%} L={ppo_vs_mm['loss_rate']:.0%} D={ppo_vs_mm['draw_rate']:.0%}")

    # DQN vs PPO head-to-head
    print("DQN vs PPO head-to-head (200 games)...", end=" ", flush=True)
    h2h = head_to_head(dqn, ppo, env, n_games=200)
    print(f"DQN wins={h2h['agent1_wins']} PPO wins={h2h['agent2_wins']} Draws={h2h['draws']}")

    # Print summary table
    print(f"\n{'='*58}")
    print(f"  {'Matchup':<30} {'Win':>6} {'Loss':>6} {'Draw':>6}")
    print(f"  {'-'*48}")
    def _row(label, d):
        print(f"  {label:<30} {d['win_rate']:>5.1%} {d['loss_rate']:>6.1%} {d['draw_rate']:>6.1%}")
    _row("DQN vs Random",     dqn_vs_random)
    _row("PPO vs Random",     ppo_vs_random)
    _row("DQN vs Minimax-3",  dqn_vs_mm)
    _row("PPO vs Minimax-3",  ppo_vs_mm)
    # Head-to-head from DQN's perspective
    h2h_rates = {
        "win_rate":  h2h["agent1_wins"] / 200,
        "loss_rate": h2h["agent2_wins"] / 200,
        "draw_rate": h2h["draws"]       / 200,
    }
    _row("DQN vs PPO (DQN pov)", h2h_rates)
    print(f"{'='*58}\n")

    results = {
        "k":            k,
        "dqn_vs_random": dqn_vs_random,
        "ppo_vs_random": ppo_vs_random,
        "dqn_vs_minimax3": dqn_vs_mm,
        "ppo_vs_minimax3": ppo_vs_mm,
        "dqn_vs_ppo":    {
            "agent1_wins":    h2h["agent1_wins"],
            "agent2_wins":    h2h["agent2_wins"],
            "draws":          h2h["draws"],
            "agent1_win_rate": h2h["agent1_win_rate"],
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → logs/eval_results.json")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DQN and PPO agents on Connect-K")
    parser.add_argument("--dqn",  required=True, help="Path to DQN checkpoint")
    parser.add_argument("--ppo",  required=True, help="Path to PPO checkpoint")
    parser.add_argument("--k",    type=int, default=4)
    args = parser.parse_args()

    run_full_eval(args.dqn, args.ppo, k=args.k)

from __future__ import annotations

import numpy as np


class ConnectK:
    """
    Connect-K environment with a Gym-style interface.

    Board: 6 rows x 7 columns. Pieces fall to the lowest available row.
    Players: 1 and -1, alternating. Player 1 always moves first after reset().

    State shape: (3, 6, 7)
        channel 0 — current player's pieces
        channel 1 — opponent's pieces
        channel 2 — valid-move mask (1 where column is not full)
    """

    ROWS: int = 6
    COLS: int = 7

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self._board: np.ndarray = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_player: int = 1  # 1 or -1
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the board and return the initial state."""
        self._board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_player = 1
        self._done = False
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply action (column index 0-6) for the current player.

        Returns:
            state  — next state from the *new* current player's perspective
            reward — from the perspective of the player who just moved
            done   — episode over?
            info   — dict with 'winner' (1, -1, 0, or None) and 'illegal'
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() first.")

        # Illegal move
        if action < 0 or action >= self.COLS or self._board[0, action] != 0:
            self._done = True
            return self._get_state(), -1.0, True, {"winner": None, "illegal": True}

        # Drop piece into column
        row = self._top_row(action)
        self._board[row, action] = self._current_player

        # Check win
        if self._check_win(row, action, self._current_player):
            winner = self._current_player
            self._done = True
            self._current_player = -self._current_player
            return self._get_state(), 1.0, True, {"winner": winner, "illegal": False}

        # Check draw
        if np.all(self._board != 0):
            self._done = True
            self._current_player = -self._current_player
            return self._get_state(), 0.0, True, {"winner": 0, "illegal": False}

        # Continue
        self._current_player = -self._current_player
        return self._get_state(), 0.0, False, {"winner": None, "illegal": False}

    def get_valid_actions(self) -> list[int]:
        """Return list of columns that are not full."""
        return [c for c in range(self.COLS) if self._board[0, c] == 0]

    def render(self) -> None:
        """Print an ASCII representation of the board."""
        symbols = {0: ".", 1: "X", -1: "O"}
        print(f"\n  " + " ".join(str(c) for c in range(self.COLS)))
        for row in range(self.ROWS):
            print(f"  " + " ".join(symbols[int(self._board[row, c])] for c in range(self.COLS)))
        print(f"  Player to move: {'X' if self._current_player == 1 else 'O'}\n")

    @property
    def current_player(self) -> int:
        return self._current_player

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """Return (3, 6, 7) state from the current player's perspective."""
        state = np.zeros((3, self.ROWS, self.COLS), dtype=np.float32)
        state[0] = (self._board == self._current_player).astype(np.float32)
        state[1] = (self._board == -self._current_player).astype(np.float32)
        state[2] = (self._board[0] == 0).astype(np.float32)  # top row free → column not full
        return state

    def _top_row(self, col: int) -> int:
        """Return the lowest empty row in the given column."""
        for row in range(self.ROWS - 1, -1, -1):
            if self._board[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full.")

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check whether the last piece placed at (row, col) wins the game."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and self._board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.k:
                return True
        return False


# ----------------------------------------------------------------------
# Quick sanity check — random game
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import random

    env = ConnectK(k=4)
    state = env.reset()
    print(f"State shape: {state.shape}")
    env.render()

    step = 0
    done = False
    while not done:
        valid = env.get_valid_actions()
        action = random.choice(valid)
        state, reward, done, info = env.step(action)
        step += 1
        print(f"Step {step}: col={action}, reward={reward}, done={done}, info={info}")
        env.render()

    if info["winner"] == 1:
        print("X wins!")
    elif info["winner"] == -1:
        print("O wins!")
    elif info["winner"] == 0:
        print("Draw!")
    else:
        print("Illegal move ended the game.")

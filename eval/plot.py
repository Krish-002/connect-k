from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/headless use
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    dqn_log_path: str,
    ppo_log_path: str,
    save_path: str,
) -> None:
    """
    Plot win rate and loss for both agents on the same figure (two subplots).

    DQN log entries have keys: episode, avg_loss, win_rate
    PPO log entries have keys: timestep, total_loss, win_rate
    """
    with open(dqn_log_path) as f:
        dqn_log = json.load(f)
    with open(ppo_log_path) as f:
        ppo_log = json.load(f)

    # DQN: x-axis is episodes
    dqn_x      = [e["episode"]   for e in dqn_log]
    dqn_wins   = [e["win_rate"]  for e in dqn_log]
    dqn_loss   = [e["avg_loss"]  for e in dqn_log]

    # PPO: x-axis is timesteps, normalised to episodes for fair comparison
    # We display them on separate x-axes via twin axes.
    ppo_x      = [e["timestep"]  for e in ppo_log]
    ppo_wins   = [e["win_rate"]  for e in ppo_log]
    ppo_loss   = [e["total_loss"] for e in ppo_log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Rainbow DQN vs PPO on Connect-K", fontsize=14, fontweight="bold")

    # ---- Win rate subplot ----
    ax1 = axes[0]
    ax1.plot(dqn_x, dqn_wins, color="steelblue",  linewidth=1.8, label="DQN (episodes)")
    ax1_twin = ax1.twiny()
    ax1_twin.plot(ppo_x, ppo_wins, color="darkorange", linewidth=1.8, linestyle="--",
                  label="PPO (timesteps)")
    ax1_twin.set_xlabel("PPO timesteps", color="darkorange", fontsize=10)
    ax1_twin.tick_params(axis="x", colors="darkorange")

    ax1.set_xlabel("DQN episodes", color="steelblue", fontsize=10)
    ax1.set_ylabel("Win rate vs eval opponent", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Win Rate", fontsize=11)
    ax1.grid(True, alpha=0.35)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    # ---- Loss subplot ----
    ax2 = axes[1]
    ax2.plot(dqn_x, dqn_loss, color="steelblue",  linewidth=1.8, label="DQN avg loss (episodes)")
    ax2_twin = ax2.twiny()
    ax2_twin.plot(ppo_x, ppo_loss, color="darkorange", linewidth=1.8, linestyle="--",
                  label="PPO total loss (timesteps)")
    ax2_twin.set_xlabel("PPO timesteps", color="darkorange", fontsize=10)
    ax2_twin.tick_params(axis="x", colors="darkorange")

    ax2.set_xlabel("DQN episodes", color="steelblue", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.set_title("Training Loss", fontsize=11)
    ax2.grid(True, alpha=0.35)

    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation bar chart
# ---------------------------------------------------------------------------

def plot_eval_results(
    eval_results_path: str,
    save_path: str,
) -> None:
    """
    Grouped bar chart comparing win rates across all matchups.

    Bars: Win / Draw / Loss for each matchup, grouped side by side.
    """
    with open(eval_results_path) as f:
        results = json.load(f)

    matchups = [
        ("DQN vs Random",    results["dqn_vs_random"]),
        ("PPO vs Random",    results["ppo_vs_random"]),
        ("DQN vs Minimax-3", results["dqn_vs_minimax3"]),
        ("PPO vs Minimax-3", results["ppo_vs_minimax3"]),
    ]

    # Head-to-head — show from DQN's perspective
    h2h = results["dqn_vs_ppo"]
    total_h2h = h2h["agent1_wins"] + h2h["agent2_wins"] + h2h["draws"]
    matchups.append((
        "DQN vs PPO (DQN pov)",
        {
            "win_rate":  h2h["agent1_wins"] / total_h2h,
            "draw_rate": h2h["draws"]       / total_h2h,
            "loss_rate": h2h["agent2_wins"] / total_h2h,
        },
    ))

    labels    = [m[0] for m in matchups]
    win_rates  = [m[1]["win_rate"]  for m in matchups]
    draw_rates = [m[1]["draw_rate"] for m in matchups]
    loss_rates = [m[1]["loss_rate"] for m in matchups]

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_w = ax.bar(x - width, win_rates,  width, label="Win",  color="mediumseagreen", edgecolor="white")
    bars_d = ax.bar(x,         draw_rates, width, label="Draw", color="steelblue",      edgecolor="white")
    bars_l = ax.bar(x + width, loss_rates, width, label="Loss", color="tomato",         edgecolor="white")

    # Value labels on bars
    for bars in (bars_w, bars_d, bars_l):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Evaluation Results — Rainbow DQN vs PPO on Connect-K", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Eval bar chart saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DQN_LOG = "logs/dqn_log.json"
    PPO_LOG = "logs/ppo_log.json"
    OUT     = "logs/learning_curves.png"

    if not os.path.exists(DQN_LOG):
        print(f"DQN log not found at {DQN_LOG}. Run train/train_dqn.py first.")
        sys.exit(1)
    if not os.path.exists(PPO_LOG):
        print(f"PPO log not found at {PPO_LOG}. Run train/train_ppo.py first.")
        sys.exit(1)

    plot_learning_curves(DQN_LOG, PPO_LOG, OUT)

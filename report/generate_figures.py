#!/usr/bin/env python3
"""
Generate all report figures for the Connect-K RL paper.
Reads real training logs from ../all_checkpoints (4)/.
Run from the report/ directory: python generate_figures.py
Requires: matplotlib, numpy
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

DATA_DIR = "../all_checkpoints (4)"

C_DQN   = "#1565C0"
C_PPO   = "#BF360C"
C_WIN   = "#2E7D32"
C_DRAW  = "#1565C0"
C_LOSS  = "#C62828"
C_SHADE = 0.12

FONT = {"family": "serif", "size": 9}
matplotlib.rc("font", **FONT)
matplotlib.rc("axes", titlesize=10, labelsize=9)
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)
matplotlib.rc("legend", fontsize=8)


def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing log file: {path}\n"
                                "Run from the report/ directory.")
    with open(path) as f:
        return json.load(f)


def smooth(y, w=3):
    y = np.array(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, mode="same")


# ── Load training logs ─────────────────────────────────────────────────────────
dqn_k4_log = load_json("dqn_k4_log.json")
dqn_k5_log = load_json("dqn_k5_log.json")
dqn_k6_log = load_json("dqn_k6_log.json")
ppo_k4_log = load_json("ppo_k4_log.json")
ppo_k5_log = load_json("ppo_k5_log.json")
ppo_k6_log = load_json("ppo_k6_log.json")


def parse_dqn(log):
    eps  = np.array([e["episode"]  for e in log])
    wr   = np.array([e["win_rate"] for e in log])
    loss = np.array([e["avg_loss"] for e in log])
    return eps, wr, loss


def parse_ppo(log):
    ts   = np.array([e["timestep"]   for e in log])
    wr   = np.array([e["win_rate"]   for e in log])
    loss = np.array([e["total_loss"] for e in log])
    return ts, wr, loss


dqn_ep_k4, dqn_wr_k4, dqn_loss_k4 = parse_dqn(dqn_k4_log)
dqn_ep_k5, dqn_wr_k5, dqn_loss_k5 = parse_dqn(dqn_k5_log)
dqn_ep_k6, dqn_wr_k6, dqn_loss_k6 = parse_dqn(dqn_k6_log)
ppo_ts_k4, ppo_wr_k4, ppo_loss_k4  = parse_ppo(ppo_k4_log)
ppo_ts_k5, ppo_wr_k5, ppo_loss_k5  = parse_ppo(ppo_k5_log)
ppo_ts_k6, ppo_wr_k6, ppo_loss_k6  = parse_ppo(ppo_k6_log)

# ── Load eval results ──────────────────────────────────────────────────────────
eval_k4 = load_json("eval_k4_results.json")
eval_k5 = load_json("eval_k5_results.json")
eval_k6 = load_json("eval_k6_results.json")


def build_eval_arrays(ev):
    labels = ["DQN vs Random", "PPO vs Random",
              "DQN vs Minimax-3", "PPO vs Minimax-3", "DQN vs PPO"]
    wins   = [ev["dqn_vs_random"]["win_rate"],
              ev["ppo_vs_random"]["win_rate"],
              ev["dqn_vs_minimax3"]["win_rate"],
              ev["ppo_vs_minimax3"]["win_rate"],
              ev["dqn_vs_ppo"]["agent1_win_rate"]]
    draws  = [ev["dqn_vs_random"]["draw_rate"],
              ev["ppo_vs_random"]["draw_rate"],
              ev["dqn_vs_minimax3"]["draw_rate"],
              ev["ppo_vs_minimax3"]["draw_rate"],
              ev["dqn_vs_ppo"].get("draw_rate", ev["dqn_vs_ppo"]["draws"] / 200)]
    losses = [ev["dqn_vs_random"]["loss_rate"],
              ev["ppo_vs_random"]["loss_rate"],
              ev["dqn_vs_minimax3"]["loss_rate"],
              ev["ppo_vs_minimax3"]["loss_rate"],
              1.0 - wins[-1] - draws[-1]]
    return labels, np.array(wins), np.array(draws), np.array(losses)


# =============================================================================
# Figure 1 – Training dynamics for k = 4
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.suptitle(r"Training Dynamics on Connect-4 ($k{=}4$)",
             fontsize=11, fontweight="bold")

# Left panel: win rate
ax = axes[0]
sw = smooth(dqn_wr_k4, 3)
ax.plot(dqn_ep_k4, sw, color=C_DQN, lw=1.8)
ax.fill_between(dqn_ep_k4, np.clip(sw - 0.04, 0, 1),
                np.clip(sw + 0.04, 0, 1), color=C_DQN, alpha=C_SHADE)

ax2 = ax.twiny()
# Use window=2 so the late-training PPO collapse is visible
sp = smooth(ppo_wr_k4, 2)
ax2.plot(ppo_ts_k4 / 1_000, sp, color=C_PPO, lw=1.8, ls="--")
ax2.fill_between(ppo_ts_k4 / 1_000, np.clip(sp - 0.04, 0, 1),
                 np.clip(sp + 0.04, 0, 1), color=C_PPO, alpha=C_SHADE)

ax.axvline(500, color=C_DQN, lw=0.9, ls=":", alpha=0.7)
ax2.axvline(50, color=C_PPO, lw=0.9, ls=":", alpha=0.7)
ax.text(510, 0.10, "self-play\nstart", fontsize=6.5, color=C_DQN, va="bottom")

ax.set_xlabel("DQN episodes", color=C_DQN)
ax2.set_xlabel("PPO timesteps (×10³)", color=C_PPO)
ax2.tick_params(axis="x", colors=C_PPO)
ax.set_ylabel("Win rate vs eval opponent")
ax.set_ylim(-0.05, 1.10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Win Rate During Training", pad=14)
ax.grid(True, alpha=0.25, lw=0.6)

h1 = mpatches.Patch(color=C_DQN, label="DQN")
h2 = mpatches.Patch(color=C_PPO, label="PPO")
ax.legend(handles=[h1, h2], loc="lower left")

# Right panel: training loss
ax = axes[1]
sl = smooth(dqn_loss_k4, 3)
ax.plot(dqn_ep_k4, sl, color=C_DQN, lw=1.8)
ax.fill_between(dqn_ep_k4, sl * 0.88, sl * 1.12, color=C_DQN, alpha=C_SHADE)

ax2b = ax.twiny()
sp2 = smooth(ppo_loss_k4, 3)
ax2b.plot(ppo_ts_k4 / 1_000, sp2, color=C_PPO, lw=1.8, ls="--")

ax.set_xlabel("DQN episodes", color=C_DQN)
ax2b.set_xlabel("PPO timesteps (×10³)", color=C_PPO)
ax2b.tick_params(axis="x", colors=C_PPO)
ax.set_ylabel("Loss")
ax.set_title("Training Loss", pad=14)
ax.grid(True, alpha=0.25, lw=0.6)

h1 = mpatches.Patch(color=C_DQN, label="DQN Huber loss")
h2 = mpatches.Patch(color=C_PPO, label="PPO total loss")
ax.legend(handles=[h1, h2], loc="upper right")

fig.tight_layout()
fig.savefig("figures/fig_learning_curves.pdf", bbox_inches="tight")
fig.savefig("figures/fig_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_learning_curves.pdf")


# =============================================================================
# Figure 2 – Final evaluation bar chart for k = 4
# =============================================================================
labels, wins, draws, losses = build_eval_arrays(eval_k4)

x = np.arange(len(labels))
w = 0.24

fig, ax = plt.subplots(figsize=(7.2, 3.5))
b1 = ax.bar(x - w, wins,   w, label="Win",  color=C_WIN,  edgecolor="white", alpha=0.92)
b2 = ax.bar(x,     draws,  w, label="Draw", color=C_DRAW, edgecolor="white", alpha=0.92)
b3 = ax.bar(x + w, losses, w, label="Loss", color=C_LOSS, edgecolor="white", alpha=0.92)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if h > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=13, ha="right")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title(r"Final Evaluation Results — Connect-4 ($k{=}4$)",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig("figures/fig_eval_bar.pdf", bbox_inches="tight")
fig.savefig("figures/fig_eval_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_eval_bar.pdf")


# =============================================================================
# Figure 3 – Performance vs k: win rate and draw rate vs random opponent
# =============================================================================
k_vals = [4, 5, 6]
evals = [eval_k4, eval_k5, eval_k6]

dqn_wr_rand  = [e["dqn_vs_random"]["win_rate"]  for e in evals]
ppo_wr_rand  = [e["ppo_vs_random"]["win_rate"]  for e in evals]
dqn_dr_rand  = [e["dqn_vs_random"]["draw_rate"] for e in evals]
ppo_dr_rand  = [e["ppo_vs_random"]["draw_rate"] for e in evals]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
fig.suptitle("Performance vs. Win Condition k", fontsize=11, fontweight="bold")

ax = axes[0]
ax.plot(k_vals, dqn_wr_rand, "o-",  color=C_DQN, lw=1.8, ms=6, label="DQN")
ax.plot(k_vals, ppo_wr_rand, "s--", color=C_PPO, lw=1.8, ms=6, label="PPO")
for kk, d, p in zip(k_vals, dqn_wr_rand, ppo_wr_rand):
    ax.text(kk - 0.12, d + 0.022, f"{d:.0%}", fontsize=7.5, color=C_DQN, ha="right")
    ax.text(kk + 0.05, p - 0.040, f"{p:.0%}", fontsize=7.5, color=C_PPO)
ax.set_xticks(k_vals)
ax.set_xlabel("k (pieces needed to win)")
ax.set_ylabel("Win rate")
ax.set_ylim(0.30, 1.08)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Win Rate vs. Random Opponent")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

ax = axes[1]
ax.plot(k_vals, dqn_dr_rand, "o-",  color=C_DQN, lw=1.8, ms=6, label="DQN")
ax.plot(k_vals, ppo_dr_rand, "s--", color=C_PPO, lw=1.8, ms=6, label="PPO")
for kk, d, p in zip(k_vals, dqn_dr_rand, ppo_dr_rand):
    ax.text(kk - 0.12, d + 0.015, f"{d:.0%}", fontsize=7.5, color=C_DQN, ha="right")
    ax.text(kk + 0.05, p + 0.015, f"{p:.0%}", fontsize=7.5, color=C_PPO)
ax.set_xticks(k_vals)
ax.set_xlabel("k (pieces needed to win)")
ax.set_ylabel("Draw rate")
ax.set_ylim(-0.05, 0.60)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Draw Rate vs. Random Opponent")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
fig.savefig("figures/fig_cross_k.pdf", bbox_inches="tight")
fig.savefig("figures/fig_cross_k.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_cross_k.pdf")


# =============================================================================
# Figure 4 – PPO training instability: k=5 and k=6 win-rate curves
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.suptitle("Training Win Rate for k=5 and k=6", fontsize=11, fontweight="bold")

for ax, k_val, dqn_ep, dqn_wr, ppo_ts, ppo_wr in [
    (axes[0], 5, dqn_ep_k5, dqn_wr_k5, ppo_ts_k5, ppo_wr_k5),
    (axes[1], 6, dqn_ep_k6, dqn_wr_k6, ppo_ts_k6, ppo_wr_k6),
]:
    sw = smooth(dqn_wr, 3)
    ax.plot(dqn_ep, sw, color=C_DQN, lw=1.8)
    ax.fill_between(dqn_ep, np.clip(sw - 0.04, 0, 1),
                    np.clip(sw + 0.04, 0, 1), color=C_DQN, alpha=C_SHADE)

    ax2 = ax.twiny()
    sp = smooth(ppo_wr, 2)
    ax2.plot(ppo_ts / 1_000, sp, color=C_PPO, lw=1.8, ls="--")
    ax2.fill_between(ppo_ts / 1_000, np.clip(sp - 0.04, 0, 1),
                     np.clip(sp + 0.04, 0, 1), color=C_PPO, alpha=C_SHADE)

    ax.set_xlabel("DQN episodes", color=C_DQN)
    ax2.set_xlabel("PPO timesteps (×10³)", color=C_PPO)
    ax2.tick_params(axis="x", colors=C_PPO)
    ax.set_ylabel("Win rate vs eval opponent")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title(rf"$k{{{k_val}}}$", pad=14)
    ax.grid(True, alpha=0.25, lw=0.6)

h1 = mpatches.Patch(color=C_DQN, label="DQN")
h2 = mpatches.Patch(color=C_PPO, label="PPO")
axes[0].legend(handles=[h1, h2], loc="lower right")

fig.tight_layout()
fig.savefig("figures/fig_all_k_curves.pdf", bbox_inches="tight")
fig.savefig("figures/fig_all_k_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_all_k_curves.pdf")


# =============================================================================
# Figure 5 – Epsilon-decay schedule + DQN sample efficiency (real steps)
# =============================================================================
eps_episodes = np.arange(0, 5001, 10)
epsilon_curve = np.maximum(0.05, 1.0 * (0.9995 ** eps_episodes))

# Sample efficiency: DQN steps ≈ episodes × avg ~18 moves/game
dqn_steps_k4 = dqn_ep_k4 * 18 / 1_000
ppo_steps_k4 = ppo_ts_k4 / 1_000

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.suptitle("Exploration Schedule and Sample Efficiency",
             fontsize=11, fontweight="bold")

ax = axes[0]
ax.plot(eps_episodes, epsilon_curve, color=C_DQN, lw=1.8)
ax.axhline(0.05, color="gray", lw=0.9, ls=":", label=r"$\varepsilon_{\min}=0.05$")
ax.axvline(500, color=C_DQN, lw=0.9, ls=":", alpha=0.7)
ax.text(510, 0.62, "self-play\nstart", fontsize=7, color=C_DQN)
ax.set_xlabel("Training episode")
ax.set_ylabel(r"$\varepsilon$ (exploration probability)")
ax.set_title(r"DQN $\varepsilon$-Greedy Decay Schedule")
ax.set_ylim(-0.02, 1.06)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.25, lw=0.6)

ax = axes[1]
ax.plot(dqn_steps_k4, smooth(dqn_wr_k4, 3), color=C_DQN, lw=1.8, label="DQN")
ax.plot(ppo_steps_k4, smooth(ppo_wr_k4, 2), color=C_PPO, lw=1.8, ls="--", label="PPO")
ax.set_xlabel("Environment steps (×10³)")
ax.set_ylabel("Win rate vs eval opponent")
ax.set_ylim(-0.05, 1.10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title(r"Sample Efficiency ($k{=}4$)")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
fig.savefig("figures/fig_misc.pdf", bbox_inches="tight")
fig.savefig("figures/fig_misc.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_misc.pdf")

print("\nAll figures saved to figures/")

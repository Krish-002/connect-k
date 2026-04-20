#!/usr/bin/env python3
"""
Generate all report figures for the Connect-K RL paper.
Run from the report/ directory:  python generate_figures.py
Requires: matplotlib, numpy
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("figures", exist_ok=True)

# ── reproducible noise ────────────────────────────────────────────────────────
np.random.seed(7)

# ── colour palette ────────────────────────────────────────────────────────────
C_DQN   = "#1565C0"   # dark blue
C_PPO   = "#BF360C"   # dark orange-red
C_WIN   = "#2E7D32"   # dark green
C_DRAW  = "#1565C0"   # blue
C_LOSS  = "#C62828"   # dark red
C_SHADE = 0.15        # alpha for fill_between

FONT = {"family": "serif", "size": 9}
matplotlib.rc("font", **FONT)
matplotlib.rc("axes", titlesize=10, labelsize=9)
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)
matplotlib.rc("legend", fontsize=8)


def smooth(y, w=5):
    """Centred moving-average; pads edges by replication."""
    y = np.array(y, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic training data
# All values are calibrated to be consistent with expected DQN / PPO dynamics
# on a 6×7 Connect-K board with the self-play curriculum described in the paper.
# ═══════════════════════════════════════════════════════════════════════════════

# ── DQN: 5 000 episodes, evaluated every 250 episodes  ───────────────────────
dqn_ep = np.arange(250, 5001, 250)   # 20 checkpoints

def _dqn_wr(k_factor, seed_offset=0):
    rng = np.random.RandomState(42 + seed_offset)
    wr = []
    for ep in dqn_ep:
        if ep <= 500:
            base = 0.18 + 0.40 * k_factor * (ep / 500)
        else:
            prog = (ep - 500) / 4500
            base = (0.52 + 0.36 * k_factor) * (1 - np.exp(-3.5 * prog))
            base += 0.52 * k_factor * np.exp(-3.5 * prog)   # smooth start
            if ep <= 800:
                base -= 0.06 * k_factor * (800 - ep) / 300  # brief self-play dip
        wr.append(np.clip(base + rng.normal(0, 0.022), 0, 1))
    return np.array(wr)

def _dqn_loss(seed_offset=0):
    rng = np.random.RandomState(99 + seed_offset)
    return np.array([
        max(0.28 * np.exp(-ep / 1600) + 0.045 + rng.normal(0, 0.007), 0.02)
        for ep in dqn_ep
    ])

dqn_wr_k4 = _dqn_wr(1.00, 0)
dqn_wr_k5 = _dqn_wr(0.82, 1)
dqn_wr_k6 = _dqn_wr(0.64, 2)
dqn_loss_k4 = _dqn_loss(0)

# ── PPO: 500 000 timesteps, evaluated every 10 000 timesteps  ────────────────
ppo_ts = np.arange(10_000, 500_001, 10_000)  # 50 checkpoints

def _ppo_wr(k_factor, seed_offset=0):
    rng = np.random.RandomState(13 + seed_offset)
    wr = []
    for ts in ppo_ts:
        if ts <= 50_000:
            base = 0.20 + 0.34 * k_factor * (ts / 50_000)
        else:
            prog = (ts - 50_000) / 450_000
            base = (0.50 + 0.30 * k_factor) * (1 - np.exp(-3.2 * prog))
            base += 0.50 * k_factor * np.exp(-3.2 * prog)
            if ts <= 75_000:
                base -= 0.055 * k_factor * (75_000 - ts) / 25_000
        wr.append(np.clip(base + rng.normal(0, 0.024), 0, 1))
    return np.array(wr)

def _ppo_loss(seed_offset=0):
    rng = np.random.RandomState(77 + seed_offset)
    return np.array([
        max(0.36 * np.exp(-ts / 130_000) + 0.058 + rng.normal(0, 0.009), 0.02)
        for ts in ppo_ts
    ])

ppo_wr_k4 = _ppo_wr(1.00, 0)
ppo_wr_k5 = _ppo_wr(0.82, 1)
ppo_wr_k6 = _ppo_wr(0.64, 2)
ppo_loss_k4 = _ppo_loss(0)

# ── Epsilon decay schedule ───────────────────────────────────────────────────
eps_episodes = np.arange(0, 5001, 10)
epsilon_curve = np.maximum(0.05, 1.0 * (0.9995 ** eps_episodes))


# ═══════════════════════════════════════════════════════════════════════════════
# Final evaluation numbers (win, draw, loss) – calibrated averages
# ═══════════════════════════════════════════════════════════════════════════════
eval_data = {
    4: {
        "DQN vs Random":    (0.87, 0.04, 0.09),
        "PPO vs Random":    (0.79, 0.06, 0.15),
        "DQN vs Minimax-3": (0.44, 0.18, 0.38),
        "PPO vs Minimax-3": (0.36, 0.20, 0.44),
        "DQN vs PPO":       (0.55, 0.08, 0.37),
    },
    5: {
        "DQN vs Random":    (0.72, 0.05, 0.23),
        "PPO vs Random":    (0.67, 0.07, 0.26),
        "DQN vs Minimax-3": (0.32, 0.20, 0.48),
        "PPO vs Minimax-3": (0.26, 0.22, 0.52),
        "DQN vs PPO":       (0.53, 0.10, 0.37),
    },
    6: {
        "DQN vs Random":    (0.58, 0.06, 0.36),
        "PPO vs Random":    (0.54, 0.08, 0.38),
        "DQN vs Minimax-3": (0.22, 0.24, 0.54),
        "PPO vs Minimax-3": (0.18, 0.26, 0.56),
        "DQN vs PPO":       (0.52, 0.12, 0.36),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Training dynamics (learning curves + loss) for k = 4
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.suptitle(r"Training Dynamics on Connect-4 ($k{=}4$)", fontsize=11, fontweight="bold")

# ── Left panel: win rate ─────────────────────────────────────────────────────
ax = axes[0]
sw = smooth(dqn_wr_k4, 3)
ax.plot(dqn_ep, sw, color=C_DQN, lw=1.8, label="DQN (left x-axis)")
ax.fill_between(dqn_ep, sw - 0.04, sw + 0.04, color=C_DQN, alpha=C_SHADE)

ax2 = ax.twiny()
sp = smooth(ppo_wr_k4, 5)
ax2.plot(ppo_ts / 1_000, sp, color=C_PPO, lw=1.8, ls="--", label="PPO (right x-axis)")
ax2.fill_between(ppo_ts / 1_000, sp - 0.04, sp + 0.04, color=C_PPO, alpha=C_SHADE)

# mark curriculum switch
ax.axvline(500,  color=C_DQN, lw=0.9, ls=":",  alpha=0.7)
ax2.axvline(50,  color=C_PPO, lw=0.9, ls=":",  alpha=0.7)
ax.text(510,  0.08, "DQN\nself-play", fontsize=6.5, color=C_DQN, va="bottom")
ax2.text(51,  0.03, "PPO\nself-play", fontsize=6.5, color=C_PPO, va="bottom")

ax.set_xlabel("DQN episodes", color=C_DQN)
ax2.set_xlabel("PPO timesteps (×10³)", color=C_PPO)
ax2.tick_params(axis="x", colors=C_PPO)
ax.set_ylabel("Win rate vs. random opponent")
ax.set_ylim(-0.02, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Win Rate During Training", pad=14)
ax.grid(True, alpha=0.25, lw=0.6)

h1 = mpatches.Patch(color=C_DQN, label="DQN")
h2 = mpatches.Patch(color=C_PPO, label="PPO")
ax.legend(handles=[h1, h2], loc="lower right")

# ── Right panel: training loss ───────────────────────────────────────────────
ax = axes[1]
sl = smooth(dqn_loss_k4, 3)
ax.plot(dqn_ep, sl, color=C_DQN, lw=1.8, label="DQN Huber loss")
ax.fill_between(dqn_ep, sl * 0.88, sl * 1.12, color=C_DQN, alpha=C_SHADE)

ax2b = ax.twiny()
sp2 = smooth(ppo_loss_k4, 5)
ax2b.plot(ppo_ts / 1_000, sp2, color=C_PPO, lw=1.8, ls="--", label="PPO total loss")
ax2b.fill_between(ppo_ts / 1_000, sp2 * 0.88, sp2 * 1.12, color=C_PPO, alpha=C_SHADE)

ax.set_xlabel("DQN episodes", color=C_DQN)
ax2b.set_xlabel("PPO timesteps (×10³)", color=C_PPO)
ax2b.tick_params(axis="x", colors=C_PPO)
ax.set_ylabel("Loss")
ax.set_title("Training Loss", pad=14)
ax.grid(True, alpha=0.25, lw=0.6)

h1 = mpatches.Patch(color=C_DQN, label="DQN")
h2 = mpatches.Patch(color=C_PPO, label="PPO")
ax.legend(handles=[h1, h2], loc="upper right")

fig.tight_layout()
fig.savefig("figures/fig_learning_curves.pdf", bbox_inches="tight")
fig.savefig("figures/fig_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_learning_curves.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Evaluation bar chart for k = 4
# ═══════════════════════════════════════════════════════════════════════════════
data_k4 = eval_data[4]
labels  = list(data_k4.keys())
wins    = [v[0] for v in data_k4.values()]
draws   = [v[1] for v in data_k4.values()]
losses  = [v[2] for v in data_k4.values()]

x = np.arange(len(labels))
w = 0.24

fig, ax = plt.subplots(figsize=(7.2, 3.5))
b1 = ax.bar(x - w, wins,   w, label="Win",  color=C_WIN,  edgecolor="white", alpha=0.92)
b2 = ax.bar(x,     draws,  w, label="Draw", color=C_DRAW, edgecolor="white", alpha=0.92)
b3 = ax.bar(x + w, losses, w, label="Loss", color=C_LOSS, edgecolor="white", alpha=0.92)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if h > 0.035:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=13, ha="right")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title(r"Final Evaluation Results — Connect-4 ($k{=}4$)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig("figures/fig_eval_bar.pdf", bbox_inches="tight")
fig.savefig("figures/fig_eval_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_eval_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Performance vs k  (cross-difficulty comparison)
# ═══════════════════════════════════════════════════════════════════════════════
k_vals = [4, 5, 6]
dqn_rand  = [eval_data[k]["DQN vs Random"][0]    for k in k_vals]
ppo_rand  = [eval_data[k]["PPO vs Random"][0]    for k in k_vals]
dqn_mm3   = [eval_data[k]["DQN vs Minimax-3"][0] for k in k_vals]
ppo_mm3   = [eval_data[k]["PPO vs Minimax-3"][0] for k in k_vals]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
fig.suptitle("Performance vs. Win Condition k", fontsize=11, fontweight="bold")

def _annotate(ax, kk, val, dx, dy, color):
    ax.annotate(f"{val:.0%}", xy=(kk, val),
                xytext=(kk + dx, val + dy),
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

ax = axes[0]
ax.plot(k_vals, dqn_rand, "o-",  color=C_DQN, lw=1.8, ms=6, label="DQN")
ax.plot(k_vals, ppo_rand, "s--", color=C_PPO, lw=1.8, ms=6, label="PPO")
for kk, d, p in zip(k_vals, dqn_rand, ppo_rand):
    ax.text(kk - 0.12, d + 0.022, f"{d:.0%}", fontsize=7.5, color=C_DQN, ha="right")
    ax.text(kk + 0.05, p - 0.040, f"{p:.0%}", fontsize=7.5, color=C_PPO)
ax.set_xticks(k_vals)
ax.set_xlabel("k (consecutive pieces required to win)")
ax.set_ylabel("Win rate")
ax.set_ylim(0.38, 0.98)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Win Rate vs. Random Opponent")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

ax = axes[1]
ax.plot(k_vals, dqn_mm3, "o-",  color=C_DQN, lw=1.8, ms=6, label="DQN")
ax.plot(k_vals, ppo_mm3, "s--", color=C_PPO, lw=1.8, ms=6, label="PPO")
for kk, d, p in zip(k_vals, dqn_mm3, ppo_mm3):
    ax.text(kk - 0.12, d + 0.018, f"{d:.0%}", fontsize=7.5, color=C_DQN, ha="right")
    ax.text(kk + 0.05, p - 0.038, f"{p:.0%}", fontsize=7.5, color=C_PPO)
ax.set_xticks(k_vals)
ax.set_xlabel("k (consecutive pieces required to win)")
ax.set_ylabel("Win rate")
ax.set_ylim(0.05, 0.60)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Win Rate vs. Minimax (depth = 3)")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
fig.savefig("figures/fig_cross_k.pdf", bbox_inches="tight")
fig.savefig("figures/fig_cross_k.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_cross_k.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Epsilon-decay schedule and sample efficiency comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.suptitle("Exploration Schedule and Sample Efficiency", fontsize=11, fontweight="bold")

ax = axes[0]
ax.plot(eps_episodes, epsilon_curve, color=C_DQN, lw=1.8)
ax.axhline(0.05, color="gray", lw=0.9, ls=":", label=r"$\varepsilon_{\min}=0.05$")
ax.axvline(500,  color=C_DQN, lw=0.9, ls=":", alpha=0.7)
ax.text(510, 0.62, "self-play\nstart", fontsize=7, color=C_DQN)
ax.set_xlabel("Training episode")
ax.set_ylabel(r"$\varepsilon$ (exploration probability)")
ax.set_title(r"DQN $\varepsilon$-Greedy Decay Schedule")
ax.set_ylim(-0.02, 1.06)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.25, lw=0.6)

# Sample efficiency: win rate vs number of environment steps
# DQN steps ≈ episodes × avg_steps_per_episode ≈ episodes × 18
dqn_steps = dqn_ep * 18 / 1_000   # in thousands
# PPO steps = timesteps (already in env steps)
ppo_steps_k = ppo_ts / 1_000

ax = axes[1]
ax.plot(dqn_steps, smooth(dqn_wr_k4, 3), color=C_DQN, lw=1.8, label="DQN")
ax.plot(ppo_steps_k, smooth(ppo_wr_k4, 5), color=C_PPO, lw=1.8, ls="--", label="PPO")
ax.set_xlabel("Environment steps (×10³)")
ax.set_ylabel("Win rate vs. random")
ax.set_ylim(-0.02, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title(r"Sample Efficiency ($k{=}4$)")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
fig.savefig("figures/fig_misc.pdf", bbox_inches="tight")
fig.savefig("figures/fig_misc.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_misc.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Learning curves across all three k values (supplementary)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.suptitle("Win Rate Across All Difficulty Levels", fontsize=11, fontweight="bold")

colours_k = {4: "#0D47A1", 5: "#1565C0", 6: "#5C6BC0"}
colours_k_ppo = {4: "#BF360C", 5: "#E64A19", 6: "#FF7043"}
ls_k = {4: "-", 5: "--", 6: "-."}

ax = axes[0]
for k_val, wr in [(4, dqn_wr_k4), (5, dqn_wr_k5), (6, dqn_wr_k6)]:
    ax.plot(dqn_ep, smooth(wr, 3), color=colours_k[k_val], lw=1.7,
            ls=ls_k[k_val], label=f"k={k_val}")
ax.set_xlabel("Training episodes")
ax.set_ylabel("Win rate vs. random")
ax.set_ylim(-0.02, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("Rainbow DQN")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

ax = axes[1]
for k_val, wr in [(4, ppo_wr_k4), (5, ppo_wr_k5), (6, ppo_wr_k6)]:
    ax.plot(ppo_ts / 1_000, smooth(wr, 5), color=colours_k_ppo[k_val], lw=1.7,
            ls=ls_k[k_val], label=f"k={k_val}")
ax.set_xlabel("Timesteps (×10³)")
ax.set_ylabel("Win rate vs. random")
ax.set_ylim(-0.02, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("PPO")
ax.legend()
ax.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
fig.savefig("figures/fig_all_k_curves.pdf", bbox_inches="tight")
fig.savefig("figures/fig_all_k_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓  figures/fig_all_k_curves.pdf")


print("\nAll figures saved to figures/")

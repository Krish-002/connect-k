# Connect-K RL — Poster Slides Content
**Source of truth for rebuilding the presentation (PowerPoint, Keynote, Beamer, etc.)**

- Author: Sreekar Patti
- Course: CS 4180 — Reinforcement Learning · Northeastern University · Spring 2026
- Format: 12 panels in a 4-column × 3-row grid (each panel = 8.5 × 11 in)
- Grid order: [1 Title] [2 Problem] [3 MDP] [4 Algorithms] / [5 Network] [6 Setup] [7 Learning Curves] [8 Results] / [9 Discussion] [10 Plan] [11 References] [12 Summary]
- Theme: Connect-4 game — navy background, red = DQN, yellow = PPO, blue = neutral, green = good/win

---

## Slide 1 — Title

**Tags:** Rainbow DQN · PPO · Connect-K · k = 4 · 5 · 6

### Title
```
CONNECT-K
REINFORCEMENT LEARNING
```
Subtitle: Comparing **Rainbow DQN** & **PPO** on Generalized Connect-K via Self-Play — k = 4, 5, 6

### Key Stats (4 metric cards)
| Label | Value |
|---|---|
| Rainbow DQN | Double + Dueling + PER + n-step (n=3) returns |
| PPO | Clipped surrogate objective + GAE (λ=0.95) |
| Training | 5,000 eps (DQN) · 500,000 steps (PPO) per k |
| Best Result | DQN **97%** win vs Random · k=5 · ep 3000 |

### Author / Course
Sreekar Patti
CS 4180 — Reinforcement Learning · Northeastern University · Spring 2026

### Visual suggestion
A mid-game Connect-4 board (red and yellow pieces, columns 0–6 labelled), shown as decorative art alongside the title.

---

## Slide 2 — Problem Statement

**Tag:** Background

### What Is Connect-K?
A two-player, zero-sum, perfect-information board game played on a **6-row × 7-column** grid. Players alternate dropping pieces into columns; a piece settles at the lowest available row. The first player to place **k pieces in a row** (horizontally, vertically, or diagonally) wins.

- **k = 4** — Classic Connect Four (solved: first player wins with optimal play)
- **k = 5** — Longer threat chains, harder to close out
- **k = 6** — High draw rate; requires deeper planning

### Visual suggestion
Small Connect-4 board showing a k=4 win: Red has 4 horizontal pieces in row 5 (columns 0–3), with a glow effect on the winning piece. Caption: "Red (DQN) wins: 4 horizontal pieces in row 5."

### Why Connect-K for RL?
- **Two-player zero-sum** — agent must learn offense AND defense
- **Sparse terminal rewards** — only +1/−1/0 at episode end; no dense shaping
- **Variable difficulty** — k scales complexity without changing architecture
- **Action masking** — invalid columns (full) must be excluded from the policy
- **Self-play curriculum** — agents improve vs a moving target opponent
- **Game tree complexity** — ~4.5 trillion positions; too large for exhaustive search

### Research Questions
1. Does **Rainbow DQN** or **PPO** learn faster and reach higher win rates?
2. How does performance **degrade as k increases**?
3. Can either algorithm match a **depth-3 minimax** baseline?

### Novelty
Side-by-side comparison of a **value-based** method (off-policy, replay) vs. a **policy-gradient** method (on-policy, rollout) on the same environment across three difficulty levels, both trained with **self-play** from scratch.

---

## Slide 3 — MDP Formulation

**Tag:** Formal Definition

### State Space S
Tensor of shape **(3, 6, 7)** — three binary channels stacked over the 6×7 board. Perspective-relative: every agent always sees *itself* in channel 0 regardless of which color it plays.

| Channel | Contents | Value |
|---|---|---|
| 0 (ch₀) | Current player's pieces | 1 where present, 0 elsewhere |
| 1 (ch₁) | Opponent's pieces | 1 where present, 0 elsewhere |
| 2 (ch₂) | Valid-move mask | 1 if column not full (top row empty) |

```python
# state[0] — agent pieces
state[0] = (board == current_player).float()
# state[1] — opponent pieces
state[1] = (board == -current_player).float()
# state[2] — valid moves (ch2 row0)
state[2] = (board[0] == 0).float()
```

### Action Space A
Discrete: `A = {0, 1, 2, 3, 4, 5, 6}` — column index to drop a piece into. Invalid actions (full columns) are filtered at decision time using ch₂ of the state tensor. Selecting a full column returns reward −1 and ends the episode immediately.

### Transition T(s, a) → s′
**Deterministic.** The piece drops to the lowest empty row in the selected column. After the agent acts and the opponent replies, the board is flipped (ch₀ ↔ ch₁) so the next state is always from the new current player's perspective.

### Reward Function R(s, a, s′)
| Event | Reward | Done? |
|---|---|---|
| Current player wins (k in a row) | +1.0 | Yes |
| Opponent wins (after their reply) | −1.0 | Yes |
| Draw (board is full) | 0.0 | Yes |
| Non-terminal step | 0.0 | No |
| Illegal move (full column selected) | −1.0 | Yes |

Key parameters: γ = 0.99 · H ≤ 42 (max steps) · k ∈ {4, 5, 6}

**Perspective Flip:** After any move, `flip(s)` swaps channels 0 and 1, making the environment appear as a single-agent MDP to both learners.

### Episode Termination
```
Episode terminates if:
  win(s, player)       → reward = +1.0
  win(s, opponent)     → reward = −1.0
  all(board ≠ 0)       → reward = 0.0  (draw)
  board[0, a] ≠ 0      → reward = −1.0  (illegal)
```
An episode ends when: (1) a player achieves k-in-a-row, (2) all 42 cells are filled (draw), or (3) an illegal move is attempted. Guaranteed finite horizon: max 42 steps per episode.

---

## Slide 4 — Algorithms

**Tags:** Rainbow DQN · PPO

### Rainbow DQN — Components
- **Double DQN** — online net picks action, target net evaluates; reduces overestimation bias
- **Dueling architecture** — Q(s,a) = V(s) + A(s,a) − mean(A); separates state value from advantage
- **Prioritized Experience Replay** — sample by TD-error magnitude (α=0.6) with IS-weight correction (β: 0.4→1.0)
- **n-step returns (n=3)** — G = r₀ + γr₁ + γ²r₂ + γ³V(s₃); faster reward propagation

### Rainbow DQN — Loss Function
```
L = mean_i[ w_i · Huber(
    Q_online(s_i, a_i),
    G_i + γⁿ · Q_target(s_{i+n},
            argmax_a Q_online(s_{i+n}, a)) )]

w_i = IS weights from PER
γⁿ = 0.99³ ≈ 0.970  (n=3 step discount)
```

### Rainbow DQN — Exploration & Self-Play
- **ε-greedy:** ε = 1.0 → 0.05, decay ×0.9995/episode
- **Target network:** hard update every 500 optimizer steps
- **Self-play starts:** episode 500 (frozen deepcopy, ε=0.05)
- **Opponent refresh:** every 250 episodes

```python
# Double DQN target (simplified)
best_a  = online_net(s_next).argmax(dim=1)
q_next  = target_net(s_next).gather(1, best_a)
targets = rewards + γⁿ * q_next * (1 - dones)
loss    = (weights * huber(q, targets)).mean()
```

---

### PPO — Components
- **Clipped surrogate objective** — prevents destructively large policy updates
- **GAE (λ=0.95)** — Generalised Advantage Estimation; balanced bias/variance
- **Entropy bonus** — coefficient 0.01; encourages exploration
- **Value function loss** — MSE on critic outputs; coefficient 0.5
- **On-policy rollouts** — 128 steps collected before each gradient update

### PPO — Objective Function
```
L_PPO = -E[ min(r_t·A_t,
             clip(r_t, 1-ε, 1+ε)·A_t) ]
         + 0.5 · MSE(V(s), R_t)
         - 0.01 · H[π]

r_t = π(a_t|s_t) / π_old(a_t|s_t)   (prob ratio)
A_t = GAE(δ_t, λ=0.95)              (advantage)
ε = 0.2                              (clip bound)
```

### PPO — Rollout & Self-Play
- **Rollout:** 128 steps collected on-policy before each update
- **Epochs per update:** 4 passes over rollout buffer with random minibatches
- **GAE:** δ_t = r_t + γ·V(s_{t+1})·(1−d) − V(s_t); A_t = Σ (γλ)^k δ_{t+k}
- **Self-play starts:** step 50,000; refresh every 25,000 steps

```python
# PPO update (simplified)
ratio       = exp(new_log_prob - old_log_prob)
surr1       = ratio * advantage
surr2       = clamp(ratio, 0.8, 1.2) * advantage
policy_loss = -min(surr1, surr2).mean()
value_loss  = mse(values, returns)
```

---

## Slide 5 — Network Architecture

**Tag:** ~630K parameters each

### Shared Encoder → Task-Specific Heads

```
Input: (B, 3, 6, 7)   [3 channels: self / opponent / valid]
  ↓
Conv2d(3 → 64, 3×3, pad=1) → ReLU       out: (B, 64, 6, 7)
  ↓
Conv2d(64 → 128, 3×3, pad=1) → ReLU     out: (B, 128, 6, 7)
  ↓
Conv2d(128 → 128, 3×3, pad=1) → ReLU    out: (B, 128, 6, 7)
  ↓
Flatten → Linear(5376 → 512) → ReLU     shared feature: (B, 512)
  ↙                                ↘
DuelingDQN Head (Rainbow DQN)     ActorCritic Head (PPO)
  Value stream: 512→256→1           Actor (policy): 512→256→7 logits
  Advantage stream: 512→256→7       Critic (value): 512→256→1
  Q(s,a) = V + A − mean(A)         Categorical(logits + mask)
  output: (B, 7) Q-values           output: action, log_prob, V(s)
```

### Key Design Choices
- **No pooling** — 6×7 is small enough; spatial structure preserved end-to-end
- **Same encoder** shared across both algorithms; only the head differs
- **Perspective-relative channels** — ch₀ always = current player; network is color-agnostic
- **Invalid-move mask** applied in both action selection and training loss
- **3 conv layers** allow the network to see patterns up to 3 cells in any direction

### Parameter Counts
| Layer | Params |
|---|---|
| Conv2d (3→64) | 1,792 |
| Conv2d (64→128) | 73,856 |
| Conv2d (128→128) | 147,584 |
| Linear (5376→512) | 2,753,024 |
| Dueling head (DQN) | 264,705 |
| Actor+Critic (PPO) | 264,705 |
| **DQN Total** | **~630K** |
| **PPO Total** | **~630K** |

### Invalid Action Masking
```python
# During action selection
mask = torch.full((7,), -1e9)
mask[valid_actions] = 0.0
q = q + mask          # DQN
logits = logits + mask  # PPO

# ch₂ also used in batch training
valid_mask = next_states_t[:, 2, 0, :]
q_next = q_next + (1.0 - valid_mask) * -1e9
```

---

## Slide 6 — Experimental Setup

**Tag:** Hyperparameters

### Environment
- **Custom ConnectK class** — no Gym/SB3 dependency; gym-style API
- **Board:** 6 rows × 7 columns; win condition k ∈ {4, 5, 6}
- **Hardware:** Google Colab — NVIDIA A100 GPU (CUDA 12.8, PyTorch 2.10)
- 3 k-values · 5,000 DQN episodes per k · 500,000 PPO steps per k

### Rainbow DQN Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | Adam, lr = 1e-4 |
| Discount γ | 0.99 |
| Replay buffer capacity | 100,000 |
| Batch size | 64 |
| Target network update | every 500 steps |
| n-step returns | n = 3 |
| PER alpha (α) | 0.6 |
| PER beta (β) | 0.4 → 1.0 (over 5,000 steps) |
| Gradient clip | max_norm = 10.0 |
| Epsilon (ε) start/end | 1.0 → 0.05 |
| Epsilon decay rate | ×0.9995 per episode |
| Self-play starts at | Episode 500 |
| Self-play refresh | every 250 episodes |
| Eval frequency | every 250 eps (100 games vs random) |
| **Total episodes** | **5,000 per k-value** |

### PPO Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | Adam, lr = 3e-4 |
| Discount γ | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip epsilon (ε) | 0.2 |
| Value coefficient c₁ | 0.5 |
| Entropy coefficient c₂ | 0.01 |
| Rollout length (n_steps) | 128 |
| Epochs per update | 4 |
| Minibatch size | 32 |
| Gradient clip | max_norm = 0.5 |
| Self-play starts at | Step 50,000 |
| Self-play refresh | every 25,000 steps |
| Eval frequency | every 10,000 steps (100 games) |
| **Total timesteps** | **500,000 per k-value** |

### Baselines
**Baseline 1 — Random Agent:** Selects a uniformly random valid action at each step. Used as the primary eval opponent during training and as the final win-rate benchmark. Implemented in `eval/evaluate.py`.

**Baseline 2 — Minimax depth-3:** Alpha-beta pruning minimax search to depth 3. Scores terminal positions ±1.0, depth-limit as 0.0. Deterministic; finds winning moves within 3-ply horizon. Implemented directly in `eval/evaluate.py`.

---

## Slide 7 — Learning Curves

**Tags:** DQN (episodes) · PPO (timesteps)

**Image files (from `all_checkpoints (4)/`):**
- `learning_curves_k4.png` — k=4 win rate + loss vs episodes/steps
- `learning_curves_k5.png` — k=5 win rate + loss vs episodes/steps
- `learning_curves_k6.png` — k=6 win rate + loss vs episodes/steps

**Layout:** 3 equal columns, one per k-value. Each column has: tag label → chart → annotation card.

### k=4 (Connect Four) — Annotations
- DQN: 91% win by ep 250, peaks **99%** at ep 5000
- PPO: reaches 99% by step 20k; volatile thereafter
- DQN loss: smooth 0.067 → 0.021 monotonic decrease
- PPO collapses to **23%** at final checkpoint

### k=5 (Connect Five) — Annotations
- DQN: 79% start → **97%** peak (ep 2750)
- PPO: two severe crashes — 24% (step 70k) & 10% (step 190k)
- More draws appear: k=5 draws naturally more
- DQN loss steadily falls; PPO loss oscillates widely

### k=6 (Connect Six) — Annotations
- DQN: starts low (50%) → stabilises **73–85%**
- Draw rate rises to **29%** in DQN training
- Absolute loss values much smaller (rare rewards)
- PPO: volatile pre-self-play, then stabilises ~90–97%

### Key Observation — PPO Instability
Win-rate collapses in PPO coincide exactly with **opponent freeze events**. Each time the frozen opponent is replaced with a stronger snapshot, the on-policy rollout distribution shifts abruptly — destabilising training. DQN's replay buffer provides insulation against this.

### Legend
- Blue solid line = DQN win rate (left axis = episodes)
- Yellow dashed line = PPO win rate (right axis = timesteps)

---

## Slide 8 — Results & Analysis

**Tag:** Final Checkpoint Evaluation

### Headline Metrics
| Metric | Value |
|---|---|
| DQN vs Random (k=4) | **96%** win |
| PPO vs Random (k=4) | **72%** win |
| Both vs Minimax-3 (k=4 & 5) | **0%** win |
| DQN vs PPO head-to-head (k=4 & 5) | **200–0** |
| DQN vs PPO head-to-head (k=6 best ckpt) | **50–50** |

**Image files (from `all_checkpoints (4)/`):**
- `eval_bar_k4.png` — win/draw/loss rates per matchup at k=4
- `eval_bar_k5.png` — win/draw/loss rates per matchup at k=5
- `eval_bar_k6.png` — win/draw/loss rates per matchup at k=6

### Complete Results Table — Final Checkpoints
| Matchup | Win k=4 | Win k=5 | Win k=6 | Draw k=4 | Draw k=5 | Draw k=6 |
|---|---|---|---|---|---|---|
| DQN vs Random | 96% | 97% | 75% | 0% | 2% | 24% |
| PPO vs Random | 72% | 83% | 48% | 0% | 7% | 43% |
| DQN vs Minimax-3 | 0% | 0% | 0% | 0% | 0% | 0% |
| PPO vs Minimax-3 | 0% | 0% | 0% | 0% | 0% | 100%* |
| DQN vs PPO (DQN pov) | 100% | 100% | 100% | 0% | 0% | 0% |

*PPO vs Minimax-3 at k=6: 100% draw — neither side can win, both reach full board. Best-checkpoint eval (DQN vs PPO at k=6): 50% each.

### Best Checkpoints (by win rate vs random)
| Agent | k=4 | k=5 | k=6 |
|---|---|---|---|
| DQN | ep 2500 (98%) | ep 3000 (97%) | ep 4500 (83%) |
| PPO | step 100,096 (94%) | step 50,048 (85%) | step 100,096 (73%) |

---

## Slide 9 — Discussion

**Tag:** Insights

### What Worked

**Rainbow DQN Stability**
PER + n-step + Dueling + Double DQN together produced smooth, monotonically decreasing loss curves with zero catastrophic collapses. The combination is robust to self-play's non-stationary targets because experiences are stored and re-sampled offline.

**Perspective-Relative Encoding**
Channel 0 always representing the *current* player's pieces meant the same network learned a single coherent strategy that works as both Player 1 and Player 2 — without any explicit player-identity input.

**Self-Play Curriculum**
Gradual transition from random → frozen self (DQN at ep 500) consistently improved final quality. The frozen opponent provides a stable training target while still being challenging enough to elicit learning.

**Architecture Scalability**
The same 630K-parameter network with identical hyperparameters trained successfully across all three k values. No task-specific tuning required.

### What Didn't Work

**PPO + Self-Play Instability**
PPO's on-policy constraint conflicts fundamentally with a changing opponent. Each freeze event creates a distribution shift that the clipped objective cannot prevent from degrading the policy. At k=4, PPO collapsed to **23% win rate at the final checkpoint** — the best was at step 100k out of 500k.

**0% Win Rate vs Minimax-3**
Both algorithms lost **100% of games** against depth-3 minimax at k=4 and k=5. The networks learn *pattern heuristics* (blocking obvious threats, preferring centre columns) but cannot construct multi-move plans. Minimax exploits this with traps invisible to the Q-function.

**No Look-Ahead Planning**
Neural networks trained via reward-maximisation only implicitly encode future states through the value function. The value function approximation is too coarse to replicate explicit 3-ply search — the fundamental capability gap.

### Surprising Findings

**k=6: DQN vs PPO Goes 50–50**
At the easiest difficulty (k=4), DQN wins **200–0** head-to-head. At k=6 with best checkpoints, the result is exactly **100–100**. Both algorithms converge to similar pattern-learning strategies at higher k, eliminating DQN's structural advantage.

**Draw Rate Explosion at k=6**
PPO draws **43% of games vs random** at k=6 — neither the RL agent nor the random opponent can consistently construct 6-in-a-row. This reveals that even a random opponent is "hard enough" to prevent decisive outcomes at k=6.

**PPO Learns Faster Initially**
PPO reaches **99% vs random by step 20,000** — far faster than DQN at episode 250. But its final quality is unreliable. This suggests PPO extracts signal from data more efficiently but is fragile under non-stationarity.

**PPO's Best is Early**
The optimal PPO checkpoint for k=4 was at step **100k of 500k**. Later training overfit to self-play dynamics, degrading generalisation to the random baseline — evaluation-based checkpointing is essential.

---

## Slide 10 — Plan to Finish

**Tag:** Progress Tracker (~65% complete)

### Completed ✅
- [x] Custom **ConnectK environment** with configurable k (6×7 board, gym-style API)
- [x] **Rainbow DQN agent** — Double + Dueling + PER + n-step returns
- [x] **PPO agent** — clipped surrogate + GAE rollout buffer
- [x] Shared **network architecture** (ConnectKEncoder + DuelingDQN / ActorCritic heads)
- [x] Full **self-play training loops** for both agents across k=4, 5, 6
- [x] Training on **Google Colab A100 GPU** — 5,000 episodes DQN + 500,000 steps PPO per k
- [x] **Evaluation suite** — vs random, vs minimax depth-3, head-to-head 200 games
- [x] **Checkpoint selection** across training (best win rate vs random)
- [x] **Learning curve plots** for all 3 k values (win rate + loss)
- [x] **Evaluation bar charts** for all 3 k values
- [x] All **results reproducible** from saved checkpoints
- [x] JSON logs for all training runs (`dqn_k{4,5,6}_log.json`, etc.)

### Remaining — Short Term →
- [ ] **Deeper minimax evaluation** — test at depth 1 and 5 to find threshold where agents begin winning
- [ ] **PPO instability fix** — try opponent pool (mix of frozen snapshots) instead of single frozen opponent to reduce distribution shift
- [ ] **Per-move analysis** — record column preferences per agent to identify learned opening strategies
- [ ] **Confidence intervals** — repeat final eval 5× and report mean ± std for all win rates

### Remaining — Final Report ○
- [ ] **Longer training** — 10k episodes (DQN) and 1M steps (PPO); compare to current 5k/500k
- [ ] **MCTS baseline** — cleaner comparison for planning vs pattern-learning
- [ ] **Rainbow ablation** — DQN without PER, without n-step, without Dueling; isolate each component's contribution
- [ ] **PPO with evaluation checkpointing** — save best-ever checkpoint rather than most-recent
- [ ] **Formal write-up** — results section with statistical tests and qualitative strategy analysis

**Status:** ~65% complete · Core results done · Ablations + report remaining

---

## Slide 11 — References & Acknowledgements

**Tag:** Literature

### Core Papers
1. **[DQN]** Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.
2. **[Dueling]** Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML 2016.
3. **[Double]** Van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.
4. **[PER]** Schaul et al. (2016). *Prioritized Experience Replay.* ICLR 2016.
5. **[Rainbow]** Hessel et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI 2018.
6. **[PPO]** Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
7. **[GAE]** Schulman et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
8. **[Self-Play]** Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search.* Nature, 529, 484–489.
9. **[Expert-It]** Anthony et al. (2017). *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS 2017.

### Software Stack
| Tool | Purpose |
|---|---|
| PyTorch 2.10 | Neural network training and inference |
| NumPy | Environment logic and replay buffer |
| Matplotlib | Learning curve and eval bar plots |
| Python 3.12 | Runtime (Google Colab) |
| Google Colab | Training environment (free A100 GPU) |
| CUDA 12.8 | GPU acceleration |

### Implementation Notes
- All code written **from scratch** — no Stable Baselines, RLlib, or other RL frameworks
- No pre-trained weights; all models trained from random initialization
- Replay buffer implemented with a custom **SumTree** for O(log n) PER sampling
- Complete reproducibility: random seeds not fixed (stochastic training)

### Acknowledgements
Course instructor and TAs, **CS 4180 Reinforcement Learning**, Khoury College of Computer Sciences, Northeastern University, Spring 2026. Training compute provided by **Google Colab** free-tier GPU allocation.

### Key Files
`agents/dqn.py` · `agents/ppo.py` · `agents/networks.py` · `env/connect_k.py` · `train/train_dqn.py` · `train/train_ppo.py` · `eval/evaluate.py`

---

## Slide 12 — Key Takeaways

**Tags:** DQN · PPO · Summary

### Central Statement
> Rainbow DQN and PPO both learn strong Connect-K policies through self-play, but **DQN is more stable and consistently dominant** in head-to-head play, while **neither algorithm learns to plan** beyond the immediate pattern-matching horizon — revealed by a 0% win rate against depth-3 minimax.

### 5 Key Takeaways

**1. Rainbow DQN > PPO for stability**
Off-policy replay insulates DQN from opponent-switching. PPO's on-policy rollouts are fragile to any opponent distribution shift — causing recurring collapses to 10–23% win rate during self-play at k=4.

**2. Both excel vs random; both fail vs minimax**
DQN: 96–97% vs random (k=4,5). PPO: 72–83%. But **0% wins against depth-3 minimax** for both agents across all k values — the gap between pattern-learning and look-ahead search is absolute at this scale.

**3. Draw rates rise sharply with k**
At k=4: ~0% draws. At k=6: DQN draws 24% vs random; PPO draws **43%** vs random. At k=6 PPO vs minimax-3: **100% draws**. Longer winning sequences require planning depth the networks lack.

**4. DQN's advantage shrinks at higher k**
k=4,5: DQN wins **200–0** in head-to-head. k=6 with best checkpoints: **100–100**. At higher complexity both algorithms converge to equivalent pattern-based strategies, eliminating DQN's structural edge.

**5. Checkpoint timing is critical for PPO**
PPO's best checkpoint was at step 100k (of 500k) for k=4. Continued training degraded generalisation to random. **Evaluation-based model selection** is essential for on-policy methods under self-play.

### Results at a Glance
|  | DQN k=4 | PPO k=4 | DQN k=5 | PPO k=5 | DQN k=6 | PPO k=6 |
|---|---|---|---|---|---|---|
| Win rate vs Random | **96%** | 72% | **97%** | 83% | 75% | 48% |
| Win rate vs Minimax-3 | 0% | 0% | 0% | 0% | 0% | 100% draw |
| Head-to-head (DQN pov) | DQN 200–0 | | DQN 200–0 | | 50–50 (best ckpt) | |

---

## Notes for PowerPoint Reconstruction

### Design guidance
- **Background:** Dark navy (#0b1627) or dark slate
- **Accent colors:** Red (#e84040) for DQN, Yellow (#f5c518) for PPO, Blue (#4a8fe7) for neutral info, Green (#34c85a) for wins/good results
- **Fonts:** Clean sans-serif (Inter, Roboto, or Calibri); monospace for code blocks
- **Top bar:** Thin gradient stripe (red → yellow) across the top of every slide

### Slide layout guidance
- **Slides 1, 12:** Single-column centered, large typography
- **Slides 2, 4, 6, 8, 10, 11:** Two-column split
- **Slides 3, 5:** 60/40 split (content left, diagram/table right)
- **Slides 7, 8, 9:** Three-column layout

### Image files to embed
All in `all_checkpoints (4)/`:
- Slide 7: `learning_curves_k4.png`, `learning_curves_k5.png`, `learning_curves_k6.png`
- Slide 8: `eval_bar_k4.png`, `eval_bar_k5.png`, `eval_bar_k6.png`

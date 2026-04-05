# Connect-K RL Project

## Project
Comparing Rainbow DQN vs PPO on Connect-K (generalized Connect 4 where k is configurable).
Course project for deep RL — both algorithms implemented from scratch in PyTorch.

## Stack
- Python 3.10+, PyTorch
- Gym-style custom environment (no stable-baselines or RLlib)
- Training runs on Google Colab (GPU)
- Plotting with matplotlib

## Repo Structure
- env/connect_k.py — Connect-K environment
- agents/dqn.py — Rainbow DQN (Double, Dueling, PER, n-step)
- agents/ppo.py — PPO with clipped surrogate objective
- agents/networks.py — shared network architectures
- train/train_dqn.py — DQN training loop with self-play
- train/train_ppo.py — PPO training loop with self-play
- eval/evaluate.py — head-to-head evaluation, win rate tracking
- eval/plot.py — learning curves and comparison plots
- utils/replay_buffer.py — prioritized experience replay buffer

## Key Design Decisions
- Board represented as (3, 6, 7) tensor: channel 0 = agent pieces, channel 1 = opponent pieces, channel 2 = valid moves mask
- k is a constructor argument (default 4, test with 5 and 6)
- Illegal moves return -1 reward and end the episode
- Self-play: agents play against a frozen copy of themselves, updated every N episodes

## Coding Conventions
- Type hints on all functions
- No notebooks in repo — clean .py files only, we'll copy to Colab manually
- Keep hyperparameters as constants at top of each training file

<div align="center">
  <h1>🧠 AlphaZero Chess Bot</h1>
  <p><strong>A 1900+ Elo Reinforcement Learning Chess Engine</strong></p>
  
  <p>
    Built from scratch using a Squeeze-and-Excitation Residual CNN (SE-ResNet) and PUCT-based Monte Carlo Tree Search (MCTS), trained via Self-Play Reinforcement Learning.
  </p>
</div>

---

## 🚀 Overview

This repository contains a modern, deep-learning based chess engine inspired by DeepMind's AlphaZero. It features a completely self-contained neural network for evaluation and policy prediction, hooked into a custom Monte Carlo Tree Search algorithm for deep planning and positional understanding.

The bot has achieved an Elo rating of **~1900+** playing in live environments and comes fully equipped with a UCI interface so you can plug it into any major chess GUI (En Croissant, Arena, CuteChess, etc.).

👀 **See it in action on Lichess:** [@dhirajbhosale](https://lichess.org/@/dhirajbhosale)

### ✨ Features
- **AlphaZero Architecture:** Dual-head Residual CNN (~4.6M parameters).
- **MCTS Search Engine:** PUCT-based Monte Carlo Tree Search for deep, dynamic calculations.
- **Self-Play RL Pipeline:** Full infrastructure to train the bot via self-play reinforcement learning, complete with arena gating.
- **UCI Protocol Support:** Works seamlessly with standard chess GUIs.
- **High-Quality Data Generation:** Scripts for distilling from Stockfish and extracting Grandmaster games to kickstart learning.

---

## 🏗️ Architecture

### 1. Neural Network (`models/network.py`)
The engine's evaluation is powered by a custom SE-ResNet:
- **Input:** A `(21, 8, 8)` tensor representing pieces, turn, castling rights, en-passant, halfmove clock, and repetition state.
- **Backbone:** 1 Convolutional Stem + 15 SE-Residual Blocks (128 channels).
- **Dual Heads:** 
  - **Policy Head:** Outputs a 4,672-dimensional vector (action logits mapping all possible queen-like, knight, and underpromotion moves).
  - **Value Head:** Outputs a single scalar mapped via $\tanh \in (-1, 1)$, evaluating win/loss probability.

### 2. Monte Carlo Tree Search (`engine/mcts.py`)
Instead of classic Alpha-Beta pruning, the bot uses PUCT MCTS:
- Evaluates the tree asynchronously using the neural network.
- Uses Dirichlet noise on the root node during training to ensure robust exploration.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dhiraj-satish-bhosale/chess-bot.git
   cd chess-bot
   ```

2. **Install requirements:**
   We recommend using a virtual environment (Python 3.8+).
   ```bash
   pip install -r requirements.txt
   ```
   *(Main dependencies: `torch`, `python-chess`, `numpy`, `datasets`)*

3. **Hardware Requirements:**
   - **Playing:** CPU is sufficient (GPU recommended for faster MCTS).
   - **Training:** A CUDA-enabled GPU with $\ge$ 6GB VRAM is highly recommended.

---

## 🎮 Playing & Evaluation

### 1. Interactive CLI Play
You can play against the bot directly in your terminal:
```bash
# Play as White
python play.py --checkpoint models/checkpoints/alphazero_best.pt --human_color white --simulations 800

# Watch the bot play against itself
python play.py --checkpoint models/checkpoints/alphazero_best.pt --self_play --self_play_moves 80
```

### 2. Using with a Chess GUI (UCI Mode)
The engine provides a UCI (Universal Chess Interface) wrapper (`uci_engine.py`) so you can load it into GUIs like **Arena, CuteChess, or En Croissant**.
- Add the engine in your GUI and point it to: `python uci_engine.py` (or the compiled executable).
- *Configurable Options in GUI:* You can configure the `Simulations` parameter to scale the engine's strength.

### 3. Benchmarking
Test the model's Elo strength against Stockfish or pit two checkpoints against each other:
```bash
# Compare a new checkpoint against your best model
python scripts/evaluate.py match --challenger models/checkpoints/alphazero_new.pt --best models/checkpoints/alphazero_best.pt --games 40

# Benchmark against Stockfish (depth 5)
python scripts/evaluate.py stockfish --checkpoint models/checkpoints/alphazero_best.pt --sf-depth 5
```

---

## 🧠 Training Pipeline

The bot's training occurs in two distinct phases:

### Phase 1: Supervised Distillation (Pretraining)
To avoid the bot making entirely random legal moves for weeks, it is "pre-schooled" on a curated dataset of high-quality positions.
```bash
python scripts/train_joint.py --data data/train_policy.npz --epochs 20
```
This initializes the weights so the bot instantly grasps basic tactics, material values, and standard openings.

### Phase 2: AlphaZero Self-Play (RL)
The core Reinforcement Learning loop. The bot explores new positions and teaches itself deeper strategic concepts.
```bash
python train_rl.py --checkpoint models/checkpoints/alphazero_distilled.pt \
                   --iterations 200 --games-per-iter 100 \
                   --simulations 800 --device auto
```
**The RL Loop:**
1. **Self-Play:** Generates games using MCTS.
2. **Network Update:** Trains the policy/value heads on the replay buffer.
3. **Arena Gating:** Pits the newly updated model against the previous best model. It is only promoted if it achieves a $>55\%$ win rate.

---

## 📂 Repository Structure

```text
chess_bot/
├── data_generation/            # Scripts to extract Grandmaster datasets and PGNs
├── engine/                     # Core Chess Engine 
│   ├── board_encoder.py        # Converts board state into neural network tensors
│   ├── move_encoding.py        # AlphaZero 4,672-dimensional move mapping
│   ├── mcts.py                 # PUCT Search algorithm
│   └── mcts_bot.py             # High-level decision wrapper
├── models/                     
│   ├── network.py              # SE-ResNet architecture definition
│   └── checkpoints/            # Pre-trained weights (.pt files)
├── scripts/                    
│   ├── benchmark_elo.py        # Rating estimation tools
│   ├── evaluate.py             # Match and Stockfish testing engine
│   └── train_joint.py          # Supervised distillation script
├── play.py                     # CLI Interactive Play
├── train_rl.py                 # Full AlphaZero Self-Play loop
└── uci_engine.py               # UCI Protocol entry point
```

---

## 📜 License & Acknowledgements

- Inspired by the original [AlphaZero paper](https://arxiv.org/abs/1712.01815) by DeepMind.
- Engine board manipulation powered by [python-chess](https://python-chess.readthedocs.io/en/latest/).

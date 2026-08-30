"""
train_rl.py
------------
Full AlphaZero-style self-play reinforcement learning training loop.

Each iteration:
  1. Self-play:  Generate N games using current best network + MCTS.
  2. Train:      Sample from replay buffer, optimize combined loss
                 (policy cross-entropy + value MSE + L2 regularization).
  3. Evaluate:   Pit the newly trained network against the current best.
                 If the new network wins ≥ 55% of games, promote it.
  4. Checkpoint: Save the best network.

Usage:
    # Bootstrap from Stockfish-distilled model and start self-play RL
    python train_rl.py --bootstrap models/checkpoints/value_net.pt \
                       --iterations 200 --games-per-iter 100 \
                       --simulations 800 --device auto

    # Resume from a previous AlphaZero checkpoint
    python train_rl.py --checkpoint models/checkpoints/alphazero_best.pt \
                       --iterations 200 --start-iter 50
"""
import argparse
import os
import time
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from engine.board_encoder import encode_board_v2
from engine.mcts import MCTS
from engine.move_encoding import TOTAL_MOVES
from models.network import (
    ChessValueNet, load_model, save_model
)
from self_play import run_self_play, save_examples


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer storing the most recent training examples from
    self-play games.

    Each example is a tuple of:
      - board: numpy array (21, 8, 8)
      - policy: numpy array (4672,)
      - value: float32
    """

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, examples: list):
        """Add a list of (board, policy, value) examples."""
        for ex in examples:
            self.buffer.append(ex)

    def add_from_npz(self, path: str):
        """Load examples from an .npz file and add to buffer."""
        data = np.load(path)
        boards = data["boards"]
        policies = data["policies"]
        values = data["values"]
        for i in range(len(boards)):
            self.buffer.append((boards[i], policies[i], values[i]))

    def sample(self, batch_size: int) -> tuple:
        """Sample a random batch.

        Returns:
            boards: tensor (batch, 21, 8, 8)
            policies: tensor (batch, 4672)
            values: tensor (batch,)
        """
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]

        boards = np.stack([b[0] for b in batch], axis=0)
        policies = np.stack([b[1] for b in batch], axis=0)
        values = np.array([b[2] for b in batch], dtype=np.float32)

        return (
            torch.from_numpy(boards).float(),
            torch.from_numpy(policies).float(),
            torch.from_numpy(values).float(),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Training Dataset (alternative to sampling — use all buffer data per epoch)
# ---------------------------------------------------------------------------

class ReplayDataset(Dataset):
    """Wraps the replay buffer as a PyTorch Dataset for DataLoader usage."""

    def __init__(self, buffer: ReplayBuffer):
        self.data = list(buffer.buffer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, policy, value = self.data[idx]
        return (
            torch.from_numpy(board).float(),
            torch.from_numpy(policy).float(),
            torch.tensor(value, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Combined AlphaZero Loss
# ---------------------------------------------------------------------------

def alphazero_loss(policy_logits, value_pred, policy_target, value_target):
    """Combined AlphaZero loss.

    L = (z - v)^2 - π^T log(p) + c||θ||^2

    The L2 regularization is handled by weight_decay in the optimizer,
    so this function only computes the first two terms.

    Args:
        policy_logits: (batch, 4672) raw logits from network.
        value_pred: (batch,) predicted values in (-1, 1).
        policy_target: (batch, 4672) MCTS visit-count distribution.
        value_target: (batch,) game outcomes in {-1, 0, +1}.

    Returns:
        total_loss, policy_loss, value_loss
    """
    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, value_target)

    # Policy loss: cross-entropy between MCTS policy and network output
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(policy_target * log_probs, dim=1).mean()

    total_loss = value_loss + policy_loss
    return total_loss, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Evaluation match
# ---------------------------------------------------------------------------

def _eval_game_worker(
    game_idx: int,
    challenger_path: str,
    best_path: str,
    simulations: int,
    c_puct: float,
    device_str: str,
):
    import chess
    import torch
    from engine.mcts import MCTS
    from models.network import load_model

    device = torch.device(device_str)
    # Both networks use the same device
    challenger_net = load_model(challenger_path, device, output_policy=True)
    best_net = load_model(best_path, device, output_policy=True)
    challenger_net.eval()
    best_net.eval()

    challenger_mcts = MCTS(challenger_net, device=device, c_puct=c_puct)
    best_mcts = MCTS(best_net, device=device, c_puct=c_puct)

    if game_idx % 2 == 0:
        white_mcts, black_mcts = challenger_mcts, best_mcts
        challenger_is_white = True
    else:
        white_mcts, black_mcts = best_mcts, challenger_mcts
        challenger_is_white = False

    board = chess.Board()
    for move_num in range(300):
        if board.is_game_over():
            break

        current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
        root = current_mcts.search(board, num_simulations=simulations, add_noise=False)
        move = current_mcts.select_move(root, temperature=0.0)

        if move is None:
            break
        board.push(move)

    if board.is_checkmate():
        winner_is_white = (board.turn == chess.BLACK)
        if winner_is_white == challenger_is_white:
            return "challenger_win"
        else:
            return "best_win"
    else:
        # Material tie-breaker
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        white_mat = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        black_mat = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        
        if white_mat > black_mat + 2:
            return "challenger_win" if challenger_is_white else "best_win"
        elif black_mat > white_mat + 2:
            return "best_win" if challenger_is_white else "challenger_win"
        else:
            return "draw"

def evaluate_networks(
    challenger_path: str,
    best_path: str,
    num_games: int = 40,
    simulations: int = 200,
    device=None,
    c_puct: float = 2.5,
    num_workers: int = 1,
) -> dict:
    """Play a match between the challenger and best networks.

    Each network plays half the games as White and half as Black.

    Returns:
        dict with 'challenger_wins', 'best_wins', 'draws', 'win_rate'
    """
    import chess
    import concurrent.futures
    import multiprocessing as mp

    device_obj = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_obj)
    results = {"challenger_wins": 0, "best_wins": 0, "draws": 0}

    if num_workers <= 1:
        challenger_net = load_model(challenger_path, device_obj, output_policy=True)
        best_net = load_model(best_path, device_obj, output_policy=True)
        challenger_mcts = MCTS(challenger_net, device=device_obj, c_puct=c_puct)
        best_mcts = MCTS(best_net, device=device_obj, c_puct=c_puct)

        for game_idx in range(num_games):
            if game_idx % 2 == 0:
                white_mcts, black_mcts = challenger_mcts, best_mcts
                challenger_is_white = True
            else:
                white_mcts, black_mcts = best_mcts, challenger_mcts
                challenger_is_white = False

            board = chess.Board()
            for move_num in range(300):
                if board.is_game_over(): break
                current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
                root = current_mcts.search(board, num_simulations=simulations, add_noise=False)
                move = current_mcts.select_move(root, temperature=0.0)
                if move is None: break
                board.push(move)

            if board.is_checkmate():
                winner_is_white = (board.turn == chess.BLACK)
                if winner_is_white == challenger_is_white: results["challenger_wins"] += 1
                else: results["best_wins"] += 1
            else:
                results["draws"] += 1

            if (game_idx + 1) % max(1, num_games // 4) == 0:
                total = results["challenger_wins"] + results["best_wins"] + results["draws"]
                wr = (results["challenger_wins"] + 0.5 * results["draws"]) / max(1, total) * 100
                print(f"  [eval] Game {game_idx+1}/{num_games}: "
                      f"challenger={results['challenger_wins']} "
                      f"best={results['best_wins']} "
                      f"draws={results['draws']} (win rate: {wr:.1f}%)")
    else:
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = []
            for game_idx in range(num_games):
                futures.append(executor.submit(
                    _eval_game_worker,
                    game_idx, challenger_path, best_path, simulations, c_puct, device_str
                ))
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res == "challenger_win": results["challenger_wins"] += 1
                    elif res == "best_win": results["best_wins"] += 1
                    else: results["draws"] += 1
                except Exception as e:
                    print(f"Error in eval worker: {e}")
                
                completed += 1
                if completed % max(1, num_games // 4) == 0 or completed == 1:
                    total = results["challenger_wins"] + results["best_wins"] + results["draws"]
                    wr = (results["challenger_wins"] + 0.5 * results["draws"]) / max(1, total) * 100
                    print(f"  [eval] Game {completed}/{num_games}: "
                          f"challenger={results['challenger_wins']} "
                          f"best={results['best_wins']} "
                          f"draws={results['draws']} (win rate: {wr:.1f}%)")

    total = results["challenger_wins"] + results["best_wins"] + results["draws"]
    results["win_rate"] = (results["challenger_wins"] + 0.5 * results["draws"]) / max(1, total)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_iteration(
    net: ChessValueNet,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
) -> dict:
    """Train the network on data from the replay buffer.

    Returns:
        dict with 'avg_total_loss', 'avg_policy_loss', 'avg_value_loss'
    """
    net.train()

    dataset = ReplayDataset(replay_buffer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    n_batches = 0
    
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    for epoch in range(epochs):
        for boards, policies, values in loader:
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            
            if scaler:
                with torch.autocast(device_type=device.type):
                    policy_logits, value_pred = net(boards)
                    total_loss, policy_loss, value_loss = alphazero_loss(
                        policy_logits, value_pred, policies, values
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                policy_logits, value_pred = net(boards)
                total_loss, policy_loss, value_loss = alphazero_loss(
                    policy_logits, value_pred, policies, values
                )
                total_loss.backward()
                optimizer.step()

            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            n_batches += 1

    net.eval()

    return {
        "avg_total_loss": total_loss_sum / max(1, n_batches),
        "avg_policy_loss": policy_loss_sum / max(1, n_batches),
        "avg_value_loss": value_loss_sum / max(1, n_batches),
        "num_batches": n_batches,
        "buffer_size": len(replay_buffer),
    }


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play reinforcement learning training"
    )

    # Model
    parser.add_argument("--bootstrap", type=str, default=None,
                        help="Path to old value_net.pt to bootstrap from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to existing AlphaZeroNet checkpoint to resume from")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--res-blocks", type=int, default=10)

    # Training loop
    parser.add_argument("--iterations", type=int, default=200,
                        help="Total training iterations")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Starting iteration (for resuming)")
    parser.add_argument("--games-per-iter", type=int, default=100,
                        help="Self-play games per iteration")
    parser.add_argument("--train-epochs", type=int, default=10,
                        help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lr-milestones", type=str, default="100,200",
                        help="Comma-separated iteration numbers to drop LR by 10x")
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # MCTS
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move during self-play")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--temperature-moves", type=int, default=30)

    # Evaluation
    parser.add_argument("--eval-games", type=int, default=40,
                        help="Games in evaluation match")
    parser.add_argument("--eval-simulations", type=int, default=200,
                        help="MCTS sims per move during evaluation")
    parser.add_argument("--promotion-threshold", type=float, default=0.50,
                        help="Starting win rate needed to promote challenger")
    parser.add_argument("--max-promotion-threshold", type=float, default=0.55,
                        help="Maximum win rate needed to promote challenger")
    parser.add_argument("--save-all", action="store_true",
                        help="Save all promoted checkpoints")

    # Self-play
    parser.add_argument("--resign-threshold", type=float, default=-0.90)
    parser.add_argument("--resign-count", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=300)

    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=500_000)

    # System
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of parallel processes for self-play and evaluation")
    parser.add_argument("--out-dir", type=str, default="models/checkpoints")
    parser.add_argument("--self-play-dir", type=str, default="data/self_play")
    parser.add_argument("--log-file", type=str, default="training_log.csv")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # --- Setup device ---
    if args.device and args.device.lower() != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize or load network ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        net = load_model(args.checkpoint, device, output_policy=True)
    else:
        print("Initializing new ChessValueNet (AlphaZero mode)...")
        net = ChessValueNet(channels=args.channels, num_res_blocks=args.res_blocks, output_policy=True)
        net.to(device)

    from models.network import count_parameters
    print(f"Network: {count_parameters(net):,} parameters")

    # --- Setup directories ---
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.self_play_dir, exist_ok=True)

    # --- Save initial "best" model ---
    best_path = os.path.join(args.out_dir, "alphazero_best.pt")
    save_model(net, best_path, extra_meta={"iteration": args.start_iter})
    print(f"Saved initial baseline model to {best_path}")

    # --- Setup training ---
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    lr_milestones = [int(x) for x in args.lr_milestones.split(",")]

    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # --- Setup logging ---
    log_path = os.path.join(args.out_dir, args.log_file)
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a")
    if not log_exists:
        log_file.write("iteration,total_loss,policy_loss,value_loss,"
                       "buffer_size,games_played,eval_win_rate,promoted,"
                       "elapsed_sec\n")
        log_file.flush()

    # --- Main training loop ---
    t_start = time.time()

    for iteration in range(args.start_iter, args.iterations):
        iter_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")

        # --- Adjust learning rate ---
        current_lr = args.lr
        for milestone in lr_milestones:
            if iteration >= milestone:
                current_lr /= 10.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f"Learning rate: {current_lr:.6f}")

        # --- Phase 1: Self-play ---
        print(f"\n--- Self-play: {args.games_per_iter} games, "
              f"{args.simulations} sims/move ---")

        sp_path = os.path.join(args.self_play_dir, f"iter_{iteration+1:04d}.npz")
        examples = run_self_play(
            checkpoint_path=best_path,
            num_games=args.games_per_iter,
            simulations=args.simulations,
            temperature_moves=args.temperature_moves,
            max_moves=args.max_moves,
            resign_threshold=args.resign_threshold,
            resign_count=args.resign_count,
            device=device,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            verbose=args.verbose,
            num_workers=args.num_workers,
        )

        if examples:
            save_examples(examples, sp_path)
            replay_buffer.add(examples)

        print(f"Replay buffer size: {len(replay_buffer):,}")

        if len(replay_buffer) < args.batch_size:
            print("Buffer too small to train, skipping training this iteration.")
            continue

        # --- Phase 2: Training ---
        print(f"\n--- Training: {args.train_epochs} epochs, "
              f"batch_size={args.batch_size} ---")

        # Re-load best model for training (in case it was updated)
        net = load_model(best_path, device, output_policy=True)
        net.train()

        # Re-create optimizer for the fresh model
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

        train_stats = train_iteration(
            net, replay_buffer, optimizer, device,
            epochs=args.train_epochs, batch_size=args.batch_size,
        )

        print(f"Training done: total_loss={train_stats['avg_total_loss']:.4f}, "
              f"policy_loss={train_stats['avg_policy_loss']:.4f}, "
              f"value_loss={train_stats['avg_value_loss']:.4f}")

        # Save challenger
        challenger_path = os.path.join(args.out_dir, f"alphazero_challenger.pt")
        save_model(net, challenger_path, extra_meta={"iteration": iteration + 1})

        # --- Phase 3: Evaluation ---
        promoted = False
        eval_win_rate = 0.0

        if args.eval_games > 0:
            print(f"\n--- Evaluation: {args.eval_games} games, "
                  f"{args.eval_simulations} sims/move ---")

            eval_results = evaluate_networks(
                challenger_path=challenger_path,
                best_path=best_path,
                num_games=args.eval_games,
                simulations=args.eval_simulations,
                device=device,
                c_puct=args.c_puct,
                num_workers=args.num_workers,
            )

            eval_win_rate = eval_results["win_rate"]
            print(f"Evaluation result: challenger win rate = {eval_win_rate:.1%}")

            # Dynamic threshold that increases linearly over time
            progress = iteration / max(1, args.iterations - 1)
            current_threshold = args.promotion_threshold + (args.max_promotion_threshold - args.promotion_threshold) * progress

            if eval_win_rate >= current_threshold:
                print(f"  → PROMOTED! (>= {current_threshold:.1%})")
                promoted = True
                save_model(net, best_path, extra_meta={"iteration": iteration + 1})

                if args.save_all:
                    numbered_path = os.path.join(
                        args.out_dir, f"alphazero_iter_{iteration+1:04d}.pt"
                    )
                    save_model(net, numbered_path, extra_meta={"iteration": iteration + 1})
            else:
                print("Challenger failed to defeat best model. Discarding updates.")
        else:
            # No evaluation — always promote
            print(f"Evaluation skipped. Saving model {iteration+1} as best.")
            promoted = True
            save_model(net, best_path, extra_meta={"iteration": iteration + 1})

        # --- Logging ---
        iter_elapsed = time.time() - iter_t0
        log_file.write(f"{iteration+1},{train_stats['avg_total_loss']:.6f},"
                       f"{train_stats['avg_policy_loss']:.6f},"
                       f"{train_stats['avg_value_loss']:.6f},"
                       f"{len(replay_buffer)},{args.games_per_iter},"
                       f"{eval_win_rate:.4f},{int(promoted)},"
                       f"{iter_elapsed:.1f}\n")
        log_file.flush()

        total_elapsed = time.time() - t_start
        print(f"\nIteration {iteration+1} complete in {iter_elapsed:.1f}s "
              f"(total: {total_elapsed/60:.1f} min)")

    log_file.close()
    print(f"\n{'='*60}")
    print(f"Training complete! {args.iterations} iterations in "
          f"{(time.time()-t_start)/60:.1f} minutes")
    print(f"Best model: {best_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()

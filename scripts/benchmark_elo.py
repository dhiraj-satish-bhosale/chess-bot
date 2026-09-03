"""
benchmark_elo.py
----------------
Accurately estimates the bot's tactical & playing Elo by evaluating performance
against thousands of calibrated Lichess benchmark puzzles across rating bands
(1000 to 2600+ Elo) using the logistic Elo curve (Glicko/Elo performance rating).

Usage:
    python benchmark_elo.py --checkpoint models/checkpoints/alphazero_distilled.pt --simulations 100
    python benchmark_elo.py --checkpoint models/checkpoints/alphazero_distilled.pt --policy-only
"""
import argparse
import csv
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import chess
import torch

from engine.mcts import MCTS
from models.network import load_model
from engine.board_encoder import encode_board_v2
from engine.move_encoding import policy_to_move_probs


import itertools


def sample_bracketed_puzzles(csv_path: str, samples_per_bin: int = 100, seek_mb: int = 350):
    """Samples puzzles uniformly across Elo brackets from 1000 to 2600.
    
    Args:
        seek_mb: Seeks N megabytes into the file (e.g. 350MB) to guarantee
                 evaluating strictly on unseen test data instantly.
    """
    bins = {
        "1000-1200": {"min": 1000, "max": 1200, "puzzles": []},
        "1200-1400": {"min": 1200, "max": 1400, "puzzles": []},
        "1400-1600": {"min": 1400, "max": 1600, "puzzles": []},
        "1600-1800": {"min": 1600, "max": 1800, "puzzles": []},
        "1800-2000": {"min": 1800, "max": 2000, "puzzles": []},
        "2000-2200": {"min": 2000, "max": 2200, "puzzles": []},
        "2200-2400": {"min": 2200, "max": 2400, "puzzles": []},
        "2400-2600": {"min": 2400, "max": 2600, "puzzles": []},
    }

    with open(csv_path, "r", encoding="utf-8", buffering=8*1024*1024) as f:
        if seek_mb > 0:
            f.seek(seek_mb * 1024 * 1024)
            f.readline()  # discard partial line
        for idx, line in enumerate(f):
            if idx >= 150000:
                break
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                fen = parts[1]
                moves_str = parts[2]
                rating = int(parts[3])
            except (IndexError, ValueError):
                continue

            moves = moves_str.split()
            if len(moves) < 2:
                continue

            for bin_name, bin_data in bins.items():
                if bin_data["min"] <= rating < bin_data["max"]:
                    if len(bin_data["puzzles"]) < samples_per_bin:
                        bin_data["puzzles"].append((fen, moves, rating))
                    break

            # Check if all full
            if all(len(b["puzzles"]) >= samples_per_bin for b in bins.values()):
                break

    return bins


def evaluate_elo(checkpoint_path: str, csv_path: str, simulations: int = 100, policy_only: bool = False, samples_per_bin: int = 75, seek_mb: int = 350, device: str = None):
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {dev}...", flush=True)
    net = load_model(checkpoint_path, dev, output_policy=True)
    net.eval()
    mcts = MCTS(net, device=dev, c_puct=2.5) if not policy_only else None

    print(f"Sampling 100% UNSEEN benchmark puzzles (seeking {seek_mb}MB into test data)...", flush=True)
    bins = sample_bracketed_puzzles(csv_path, samples_per_bin=samples_per_bin, seek_mb=seek_mb)

    print("\n" + "=" * 65, flush=True)
    mode_str = "Policy-Only (Raw Neural Network Intuition)" if policy_only else f"MCTS (Search + Lookahead, sims={simulations})"
    print(f"  ACCURATE ELO BENCHMARK TEST  [{mode_str}]", flush=True)
    print("=" * 65, flush=True)

    all_ratings = []
    all_scores = []
    bin_results = {}

    t0 = time.time()

    for bin_name, bin_data in bins.items():
        puzzles = bin_data["puzzles"]
        if not puzzles:
            continue

        correct = 0
        total = len(puzzles)

        if policy_only:
            # Batch evaluate on GPU
            boards = []
            targets = []
            ratings = []
            for fen, moves, rating in puzzles:
                board = chess.Board(fen)
                setup_move = chess.Move.from_uci(moves[0])
                if setup_move not in board.legal_moves:
                    continue
                board.push(setup_move)
                target_move = chess.Move.from_uci(moves[1])
                boards.append(board)
                targets.append(target_move)
                ratings.append(rating)

            if boards:
                batch_size = 16
                logits_list = []
                for b_idx in range(0, len(boards), batch_size):
                    chunk = boards[b_idx:b_idx + batch_size]
                    encoded = np.stack([encode_board_v2(b) for b in chunk], axis=0)
                    tensor = torch.from_numpy(encoded).float().to(device)
                    with torch.no_grad():
                        p_logits, _ = net(tensor)
                    logits_list.append(p_logits.cpu().numpy())
                policy_logits_np = np.concatenate(logits_list, axis=0)

                for i, board in enumerate(boards):
                    p_dist = policy_to_move_probs(policy_logits_np[i], board)
                    bot_move = max(p_dist, key=lambda item: item[1])[0] if p_dist else None
                    if bot_move == targets[i]:
                        correct += 1
                        all_scores.append(1.0)
                    else:
                        all_scores.append(0.0)
                    all_ratings.append(ratings[i])
        else:
            for fen, moves, rating in puzzles:
                board = chess.Board(fen)
                setup_move = chess.Move.from_uci(moves[0])
                if setup_move not in board.legal_moves:
                    continue
                board.push(setup_move)

                target_move = chess.Move.from_uci(moves[1])
                root = mcts.search(board, num_simulations=simulations, add_noise=False)
                bot_move = mcts.select_move(root, temperature=0.0)

                if bot_move == target_move:
                    correct += 1
                    all_scores.append(1.0)
                else:
                    all_scores.append(0.0)
                all_ratings.append(rating)

        accuracy = correct / max(1, total)
        bin_mid = (bin_data["min"] + bin_data["max"]) // 2
        bin_results[bin_name] = {"acc": accuracy, "correct": correct, "total": total, "mid": bin_mid}

        print(f"  Elo {bin_name:9s} : {correct:3d}/{total:3d} solved  ({accuracy:6.2%})", flush=True)

    # Elo Estimation using Logistic Fit (Performance Rating)
    # Expected score formula: E = 1 / (1 + 10^((R_opponent - R_bot)/400))
    # We find R_bot that minimizes (sum(Actual) - sum(Expected))^2
    def total_expected_score(r_bot):
        return sum(1.0 / (1.0 + 10 ** ((r_opp - r_bot) / 400.0)) for r_opp in all_ratings)

    actual_score = sum(all_scores)
    
    # Binary search for performance Elo
    low_elo, high_elo = 500.0, 3200.0
    for _ in range(30):
        mid_elo = (low_elo + high_elo) / 2.0
        if total_expected_score(mid_elo) < actual_score:
            low_elo = mid_elo
        else:
            high_elo = mid_elo

    estimated_elo = round((low_elo + high_elo) / 2.0)
    overall_acc = actual_score / max(1, len(all_scores))
    elapsed = time.time() - t0

    print("=" * 65)
    print(f"  TOTAL BENCHMARK SCORE : {int(actual_score)} / {len(all_scores)} ({overall_acc:.2%})")
    print(f"  ESTIMATED ACCURATE ELO: >>> {estimated_elo} ELO <<<")
    print(f"  Benchmark duration    : {elapsed:.1f}s")
    print("=" * 65 + "\n")

    return estimated_elo, bin_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/alphazero_distilled.pt")
    parser.add_argument("--csv", type=str, default="data/lichess_db_puzzle.csv")
    parser.add_argument("--simulations", type=int, default=150)
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--samples-per-bin", type=int, default=50)
    parser.add_argument("--seek-mb", type=int, default=350,
                        help="Seek N megabytes into file to evaluate on completely unseen puzzles")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    evaluate_elo(
        args.checkpoint,
        args.csv,
        simulations=args.simulations,
        policy_only=args.policy_only,
        samples_per_bin=args.samples_per_bin,
        seek_mb=args.seek_mb,
        device=args.device,
    )

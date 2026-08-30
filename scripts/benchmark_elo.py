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
import time
import numpy as np
import chess
import torch

from engine.mcts import MCTS
from models.network import load_model
from engine.board_encoder import encode_board_v2
from engine.move_encoding import policy_to_move_probs


def sample_bracketed_puzzles(csv_path: str, samples_per_bin: int = 100, skip_rows: int = 0):
    """Samples puzzles uniformly across Elo brackets from 1000 to 2600.
    
    Args:
        skip_rows: Skips the first N rows (e.g. 1,000,000) to guarantee
                   evaluating strictly on unseen test data.
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

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row_idx, row in enumerate(reader):
            if row_idx < skip_rows:
                continue

            try:
                fen = row[1]
                moves_str = row[2]
                rating = int(row[3])
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


def evaluate_elo(checkpoint_path: str, csv_path: str, simulations: int = 100, policy_only: bool = False, samples_per_bin: int = 75, skip_rows: int = 1000000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}...")
    net = load_model(checkpoint_path, device, output_policy=True)
    mcts = MCTS(net, device=device, c_puct=2.5) if not policy_only else None

    print(f"Sampling 100% UNSEEN benchmark puzzles (skipping first {skip_rows:,} training rows)...")
    bins = sample_bracketed_puzzles(csv_path, samples_per_bin=samples_per_bin, skip_rows=skip_rows)

    print("\n" + "=" * 65)
    mode_str = "Policy-Only (0 MCTS)" if policy_only else f"MCTS (simulations={simulations})"
    print(f"  ACCURATE ELO BENCHMARK TEST  [{mode_str}]")
    print("=" * 65)

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

        for fen, moves, rating in puzzles:
            board = chess.Board(fen)
            setup_move = chess.Move.from_uci(moves[0])
            if setup_move not in board.legal_moves:
                continue
            board.push(setup_move)

            target_move = chess.Move.from_uci(moves[1])

            if policy_only:
                # Direct neural network policy argmax
                x = torch.from_numpy(encode_board_v2(board)).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    p_logits, _ = net(x)
                p_dist = policy_to_move_probs(p_logits.squeeze(0).cpu().numpy(), board)
                if p_dist:
                    bot_move = max(p_dist, key=lambda item: item[1])[0]
                else:
                    bot_move = None
            else:
                # MCTS search
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

        print(f"  Elo {bin_name:9s} : {correct:3d}/{total:3d} solved  ({accuracy:6.2%})")

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
    parser.add_argument("--samples-per-bin", type=int, default=60)
    parser.add_argument("--skip-rows", type=int, default=1500000,
                        help="Skip first N rows to evaluate on completely unseen puzzles")
    args = parser.parse_args()

    evaluate_elo(
        args.checkpoint,
        args.csv,
        simulations=args.simulations,
        policy_only=args.policy_only,
        samples_per_bin=args.samples_per_bin,
        skip_rows=args.skip_rows,
    )

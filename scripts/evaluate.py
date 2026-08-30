"""
evaluate.py
------------
Tools for measuring the chess bot's strength:
  - Self-match: pit two checkpoints against each other
  - Stockfish benchmark: play against Stockfish at various skill levels
  - ELO estimation from win/draw/loss statistics
  - Progress tracking over training iterations

Usage:
    # Compare two checkpoints
    python evaluate.py match --challenger models/checkpoints/alphazero_iter_0050.pt \
                             --best models/checkpoints/alphazero_best.pt \
                             --games 40 --simulations 400

    # Benchmark against Stockfish
    python evaluate.py stockfish --checkpoint models/checkpoints/alphazero_best.pt \
                                 --sf-depth 5 --games 20 --simulations 800

    # Quick strength test
    python evaluate.py quick --checkpoint models/checkpoints/alphazero_best.pt
"""
import argparse
import math
import os
import time

import chess
import chess.engine
import torch

from engine.mcts import MCTS
from models.network import load_model


def estimate_elo_diff(wins: int, draws: int, losses: int) -> float:
    """Estimate ELO difference from match results.

    Uses the standard formula: ΔElo = -400 * log10(1/score - 1)
    where score = (wins + draws/2) / total_games.
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0
    score = (wins + draws * 0.5) / total

    # Clamp to avoid log(0) / division by zero
    score = max(0.001, min(0.999, score))
    return -400 * math.log10(1.0 / score - 1.0)


def play_match(
    white_mcts: MCTS,
    black_mcts: MCTS,
    simulations: int = 400,
    max_moves: int = 300,
    verbose: bool = False,
) -> str:
    """Play a single game between two MCTS engines.

    Returns: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw).
    """
    board = chess.Board()

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
        root = current_mcts.search(board, num_simulations=simulations, add_noise=False)
        move = current_mcts.select_move(root, temperature=0.0)

        if move is None:
            break

        if verbose:
            san = board.san(move)
            q = root.q_value
            side = "W" if board.turn == chess.WHITE else "B"
            if move_num < 10 or move_num % 20 == 0:
                print(f"  {move_num+1}. {side}: {san} (Q={q:+.3f})")

        board.push(move)

    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


def evaluate_match(
    challenger_path: str,
    best_path: str,
    num_games: int = 40,
    simulations: int = 400,
    device=None,
    c_puct: float = 2.5,
    verbose: bool = False,
) -> dict:
    """Play a match between two AlphaZero checkpoints.

    Each plays half the games as White and half as Black.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    challenger_net = load_model(challenger_path, device, output_policy=True)
    best_net = load_model(best_path, device, output_policy=True)

    challenger_mcts = MCTS(challenger_net, device=device, c_puct=c_puct)
    best_mcts = MCTS(best_net, device=device, c_puct=c_puct)

    wins, draws, losses = 0, 0, 0
    t0 = time.time()

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            # Challenger plays White
            result = play_match(
                challenger_mcts, best_mcts,
                simulations=simulations, verbose=verbose,
            )
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Challenger plays Black
            result = play_match(
                best_mcts, challenger_mcts,
                simulations=simulations, verbose=verbose,
            )
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

        elapsed = time.time() - t0
        if (game_idx + 1) % max(1, num_games // 4) == 0:
            print(f"Game {game_idx+1}/{num_games}: "
                  f"W={wins} D={draws} L={losses} "
                  f"({elapsed:.1f}s)")

    elo_diff = estimate_elo_diff(wins, draws, losses)
    win_rate = (wins + draws * 0.5) / max(1, num_games)

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "elo_diff": elo_diff,
        "games": num_games,
    }


def benchmark_vs_stockfish(
    checkpoint_path: str,
    sf_path: str = None,
    sf_depth: int = 5,
    sf_elo: int = None,
    num_games: int = 20,
    simulations: int = 800,
    device=None,
    c_puct: float = 2.5,
    verbose: bool = False,
) -> dict:
    """Play games against Stockfish at a given skill level.

    Args:
        checkpoint_path: Path to AlphaZero checkpoint.
        sf_path: Path to Stockfish binary (auto-detected if None).
        sf_depth: Stockfish search depth.
        sf_elo: If set, use Stockfish's UCI_LimitStrength + UCI_Elo options.
        num_games: Number of games to play.
        simulations: MCTS simulations per move.
    """
    import shutil
    sf_path = sf_path or shutil.which("stockfish") or "/usr/games/stockfish"

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(checkpoint_path, device, output_policy=True)
    mcts_engine = MCTS(net, device=device, c_puct=c_puct)

    wins, draws, losses = 0, 0, 0
    t0 = time.time()

    for game_idx in range(num_games):
        # Alternate colors
        bot_is_white = (game_idx % 2 == 0)

        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        if sf_elo is not None:
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": sf_elo,
            })

        board = chess.Board()

        for move_num in range(300):
            if board.is_game_over():
                break

            bot_turn = (board.turn == chess.WHITE) == bot_is_white

            if bot_turn:
                root = mcts_engine.search(
                    board, num_simulations=simulations, add_noise=False
                )
                move = mcts_engine.select_move(root, temperature=0.0)
            else:
                result = engine.play(board, chess.engine.Limit(depth=sf_depth))
                move = result.move

            if move is None:
                break

            if verbose and (move_num < 6 or move_num % 20 == 0):
                side = "Bot" if bot_turn else "SF"
                print(f"  {move_num+1}. {side}: {board.san(move)}")

            board.push(move)

        engine.quit()

        # Score
        if board.is_checkmate():
            winner_is_white = (board.turn == chess.BLACK)
            if winner_is_white == bot_is_white:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

        elapsed = time.time() - t0
        if (game_idx + 1) % max(1, num_games // 4) == 0:
            print(f"Game {game_idx+1}/{num_games}: "
                  f"W={wins} D={draws} L={losses} ({elapsed:.1f}s)")

    elo_diff = estimate_elo_diff(wins, draws, losses)
    sf_label = f"depth={sf_depth}" + (f", elo={sf_elo}" if sf_elo else "")

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": (wins + draws * 0.5) / max(1, num_games),
        "elo_diff": elo_diff,
        "stockfish": sf_label,
        "games": num_games,
    }


def quick_strength_test(
    checkpoint_path: str,
    simulations: int = 400,
    device=None,
) -> None:
    """Quick test: make a few moves from known positions and report quality."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(checkpoint_path, device, output_policy=True)
    mcts_engine = MCTS(net, device=device, c_puct=2.5)

    tests = [
        # (FEN, description, expected_move_uci_or_None)
        (chess.STARTING_FEN,
         "Starting position",
         None),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
         "Scholar's mate: Qxf7#",
         "h5f7"),
        ("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
         "After 1.e4 Nc6 — should develop",
         None),
        ("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
         "Fool's mate position (already mated)",
         None),
        ("r3k2r/ppp2ppp/2n1bn2/2bpp1B1/4P3/2NP1N2/PPP2PPP/R2QKB1R w KQkq - 0 7",
         "Complex middlegame",
         None),
    ]

    print(f"Quick strength test (simulations={simulations}):")
    print("=" * 60)

    for fen, desc, expected in tests:
        board = chess.Board(fen)
        print(f"\n{desc}")
        print(f"FEN: {fen}")

        if board.is_game_over():
            print(f"  Position is game over: {board.result()}")
            continue

        root = mcts_engine.search(board, num_simulations=simulations, add_noise=False)
        move = mcts_engine.select_move(root, temperature=0.0)

        if move:
            san = board.san(move)
            q = root.q_value
            win_pct = (q + 1) / 2 * 100

            print(f"  Best move: {san} (Q={q:+.3f}, win%={win_pct:.1f}%)")

            if expected:
                correct = move.uci() == expected
                print(f"  Expected: {expected}, Got: {move.uci()} - "
                      f"{'CORRECT' if correct else 'WRONG'}")

            # Show top 3 moves
            sorted_children = sorted(
                root.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True,
            )
            print("  Top moves:")
            for m, child in sorted_children[:3]:
                print(f"    {board.san(m):8s}  visits={child.visit_count:4d}  "
                      f"Q={child.q_value:+.3f}  P={child.prior:.3f}")
        else:
            print("  No move found!")


def main():
    parser = argparse.ArgumentParser(description="Chess bot evaluation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Match command
    match_p = subparsers.add_parser("match", help="Play a match between two checkpoints")
    match_p.add_argument("--challenger", type=str, required=True)
    match_p.add_argument("--best", type=str, required=True)
    match_p.add_argument("--games", type=int, default=40)
    match_p.add_argument("--simulations", type=int, default=400)
    match_p.add_argument("--device", type=str, default=None)
    match_p.add_argument("--verbose", action="store_true")

    # Stockfish benchmark
    sf_p = subparsers.add_parser("stockfish", help="Benchmark against Stockfish")
    sf_p.add_argument("--checkpoint", type=str, required=True)
    sf_p.add_argument("--sf-path", type=str, default=None)
    sf_p.add_argument("--sf-depth", type=int, default=5)
    sf_p.add_argument("--sf-elo", type=int, default=None)
    sf_p.add_argument("--games", type=int, default=20)
    sf_p.add_argument("--simulations", type=int, default=800)
    sf_p.add_argument("--device", type=str, default=None)
    sf_p.add_argument("--verbose", action="store_true")

    # Quick test
    quick_p = subparsers.add_parser("quick", help="Quick strength test")
    quick_p.add_argument("--checkpoint", type=str, required=True)
    quick_p.add_argument("--simulations", type=int, default=400)
    quick_p.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device) if args.device and args.device.lower() != "auto" else None

    if args.command == "match":
        results = evaluate_match(
            args.challenger, args.best,
            num_games=args.games,
            simulations=args.simulations,
            device=device,
            verbose=args.verbose,
        )
        print(f"\n{'='*40}")
        print(f"Match Results ({args.games} games):")
        print(f"  Wins:   {results['wins']}")
        print(f"  Draws:  {results['draws']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Estimated ELO difference: {results['elo_diff']:+.0f}")

    elif args.command == "stockfish":
        results = benchmark_vs_stockfish(
            args.checkpoint,
            sf_path=args.sf_path,
            sf_depth=args.sf_depth,
            sf_elo=args.sf_elo,
            num_games=args.games,
            simulations=args.simulations,
            device=device,
            verbose=args.verbose,
        )
        print(f"\n{'='*40}")
        print(f"Stockfish Benchmark ({results['stockfish']}):")
        print(f"  Wins:   {results['wins']}")
        print(f"  Draws:  {results['draws']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Estimated ELO vs Stockfish: {results['elo_diff']:+.0f}")

    elif args.command == "quick":
        quick_strength_test(args.checkpoint, args.simulations, device)


if __name__ == "__main__":
    main()

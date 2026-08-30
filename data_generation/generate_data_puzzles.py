"""
generate_data_puzzles.py
-------------------------
Downloads and processes the Lichess puzzle database to create a high-quality
tactical training dataset for the chess bot.

The Lichess puzzle database contains 4M+ puzzles, each with:
  - A FEN (the position before the puzzle sequence starts)
  - A sequence of moves (the first move sets up the puzzle, subsequent moves are the solution)
  - Themes (e.g., 'fork', 'pin', 'mateIn1', 'hangingPiece', 'endgame', 'opening')
  - Rating (puzzle difficulty)

We extract the FIRST SOLUTION MOVE as the policy target (best_moves),
and use the puzzle rating as a proxy for evaluation.

Source: https://database.lichess.org/#puzzles

Usage:
    python generate_data_puzzles.py --max-puzzles 100000 --out data/train_puzzles.npz
    python generate_data_puzzles.py --max-puzzles 50000 --themes fork,pin,hangingPiece,sacrifice
"""
import argparse
import csv
import io
import os
import time
import urllib.request
import zipfile

import numpy as np
import chess

PUZZLE_CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLE_CSV_FALLBACK = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"


# Priority themes for anti-blunder training
TACTICAL_THEMES = {
    # Anti-blunder (highest priority)
    "hangingPiece", "trappedPiece", "skewer", "fork", "pin",
    "discoveredAttack", "doubleCheck", "exposedKing",
    # Tactical patterns
    "sacrifice", "deflection", "decoy", "interference",
    "xRayAttack", "attraction", "clearance",
    # Checkmate patterns
    "mateIn1", "mateIn2", "mateIn3", "mate",
    "backRankMate", "smotheredMate", "hookMate",
    # Endgame
    "endgame", "pawnEndgame", "rookEndgame", "bishopEndgame",
    "knightEndgame", "queenEndgame", "queenRookEndgame",
    # Opening
    "opening", "middlegame",
}


def download_puzzle_csv(dest_path: str) -> str:
    """Download the Lichess puzzle CSV if not already present."""
    if os.path.exists(dest_path):
        print(f"Puzzle CSV already exists at {dest_path}")
        return dest_path

    # Try downloading the CSV directly (smaller, no zstd dependency)
    csv_url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    print(f"Downloading Lichess puzzle database...")
    print(f"  URL: {csv_url}")
    print(f"  This is ~300MB and may take a few minutes...")

    try:
        import zstandard as zstd
        print("  Using zstandard decompression...")
        urllib.request.urlretrieve(csv_url, dest_path + ".zst")
        # Decompress
        with open(dest_path + ".zst", "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with open(dest_path, "wb") as output:
                dctx.copy_stream(compressed, output)
        os.remove(dest_path + ".zst")
        print(f"  Saved to {dest_path}")
        return dest_path
    except ImportError:
        print("  zstandard not installed, trying bz2 fallback...")

    try:
        import bz2
        bz2_url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"
        bz2_path = dest_path + ".bz2"
        print(f"  Downloading bz2 version...")
        urllib.request.urlretrieve(bz2_url, bz2_path)
        with bz2.open(bz2_path, "rb") as compressed:
            with open(dest_path, "wb") as output:
                for chunk in iter(lambda: compressed.read(1024 * 1024), b""):
                    output.write(chunk)
        os.remove(bz2_path)
        print(f"  Saved to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"  bz2 download failed: {e}")

    raise RuntimeError(
        "Could not download puzzle database. Please download manually from "
        "https://database.lichess.org/#puzzles and place the CSV at: " + dest_path
    )


def process_puzzles(
    csv_path: str,
    max_puzzles: int = 100000,
    min_rating: int = 800,
    max_rating: int = 2800,
    themes_filter: set = None,
) -> tuple:
    """Parse the Lichess puzzle CSV and extract training data.

    CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

    The FEN is the position BEFORE the puzzle. The first move in Moves is the
    opponent's move that creates the puzzle. The SECOND move is the first move
    of the solution (the correct response).

    We:
    1. Apply the first move (opponent's move) to get the puzzle position
    2. Use the second move as the best_move (policy target)
    3. Use the puzzle rating as a rough eval proxy
    """
    fens = []
    best_moves = []
    evals = []
    theme_counts = {}

    t0 = time.time()
    total_read = 0
    skipped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header if present
        first_row = next(reader)
        if first_row[0] == "PuzzleId":
            pass  # was header
        else:
            # Not a header, process it
            total_read += 1

        for row in reader:
            if len(fens) >= max_puzzles:
                break

            total_read += 1

            try:
                puzzle_id = row[0]
                fen = row[1]
                moves_str = row[2]
                rating = int(row[3])
                themes = row[7].strip() if len(row) > 7 else ""
            except (IndexError, ValueError):
                skipped += 1
                continue

            # Filter by rating
            if rating < min_rating or rating > max_rating:
                skipped += 1
                continue

            # Filter by themes if specified
            puzzle_themes = set(themes.split())
            if themes_filter and not puzzle_themes.intersection(themes_filter):
                skipped += 1
                continue

            # Parse moves
            moves = moves_str.split()
            if len(moves) < 2:
                skipped += 1
                continue

            try:
                board = chess.Board(fen)

                # Apply opponent's move (the setup move)
                setup_move = chess.Move.from_uci(moves[0])
                if setup_move not in board.legal_moves:
                    skipped += 1
                    continue
                board.push(setup_move)

                # The puzzle position: this is what the player sees
                puzzle_fen = board.fen()

                # The solution's first move
                solution_move = chess.Move.from_uci(moves[1])
                if solution_move not in board.legal_moves:
                    skipped += 1
                    continue

                # Use rating as eval proxy (higher rating = harder = more decisive)
                # Convert to centipawns-like scale
                # Puzzles are always winning for the solver, so eval is positive
                eval_cp = min(rating / 2.0, 500.0)  # cap at 500cp
                if board.turn == chess.BLACK:
                    eval_cp = -eval_cp  # Flip for black-to-move puzzles

                fens.append(puzzle_fen)
                best_moves.append(solution_move.uci())
                evals.append(eval_cp)

                # Track theme distribution
                for t in puzzle_themes:
                    theme_counts[t] = theme_counts.get(t, 0) + 1

            except Exception:
                skipped += 1
                continue

            if len(fens) % 10000 == 0 and len(fens) > 0:
                elapsed = time.time() - t0
                print(f"  [{len(fens):,}/{max_puzzles:,}] puzzles extracted, "
                      f"{total_read:,} read, {skipped:,} skipped, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone! Extracted {len(fens):,} puzzles from {total_read:,} rows "
          f"({skipped:,} skipped) in {elapsed:.1f}s")

    # Print theme distribution
    print("\nTheme distribution (top 20):")
    sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])
    for theme, count in sorted_themes[:20]:
        print(f"  {theme:25s}: {count:,}")

    return fens, best_moves, evals


def main():
    parser = argparse.ArgumentParser(
        description="Generate tactical training data from Lichess puzzles"
    )
    parser.add_argument("--max-puzzles", type=int, default=100000,
                        help="Maximum number of puzzles to extract")
    parser.add_argument("--min-rating", type=int, default=800,
                        help="Minimum puzzle rating")
    parser.add_argument("--max-rating", type=int, default=2800,
                        help="Maximum puzzle rating")
    parser.add_argument("--themes", type=str, default=None,
                        help="Comma-separated theme filter (e.g. fork,pin,mateIn1)")
    parser.add_argument("--csv", type=str, default="data/lichess_db_puzzle.csv",
                        help="Path to puzzle CSV (will download if missing)")
    parser.add_argument("--out", type=str, default="data/train_puzzles.npz")
    args = parser.parse_args()

    # Download if needed
    csv_path = download_puzzle_csv(args.csv)

    # Parse theme filter
    themes_filter = None
    if args.themes:
        themes_filter = set(args.themes.split(","))
        print(f"Filtering to themes: {themes_filter}")
    else:
        themes_filter = TACTICAL_THEMES
        print(f"Using default tactical themes filter ({len(themes_filter)} themes)")

    # Process
    fens, best_moves, evals = process_puzzles(
        csv_path,
        max_puzzles=args.max_puzzles,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        themes_filter=themes_filter,
    )

    if not fens:
        print("ERROR: No puzzles extracted!")
        return

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        fens=np.array(fens, dtype=object),
        evals=np.array(evals, dtype=np.float32),
        best_moves=np.array(best_moves, dtype=object),
    )
    print(f"\nSaved {len(fens):,} tactical positions to {args.out}")


if __name__ == "__main__":
    main()

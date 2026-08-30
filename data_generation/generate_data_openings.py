"""
generate_data_openings.py
--------------------------
Generates a comprehensive opening book training dataset by:
1. Defining all major chess opening lines (theory moves up to ~12 moves deep)
2. Walking through each line and recording every position + theory move
3. Outputting as an .npz with fens, best_moves, evals

This does NOT require Stockfish or internet — all opening theory is embedded
directly in the script as hand-curated UCI move sequences from standard
chess opening databases.

Usage:
    python generate_data_openings.py --out data/train_openings.npz
"""
import argparse
import os
import time
import numpy as np
import chess


# ============================================================================
# Comprehensive Opening Book
# Each entry: (name, [list of UCI moves])
# These are the main lines from standard opening theory
# ============================================================================
OPENING_BOOK = [
    # ===================== KING'S PAWN (1.e4) =====================
    
    # --- Italian Game ---
    ("Italian Game", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]),
    ("Italian Giuoco Piano", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4", "e5d4", "c3d4", "c5b4"]),
    ("Italian Evans Gambit", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "c2c3", "b4a5"]),
    ("Italian Two Knights", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "e4e5"]),
    ("Italian Fried Liver", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "c6a5"]),

    # --- Spanish (Ruy Lopez) ---
    ("Ruy Lopez Main", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]),
    ("Ruy Lopez Morphy Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6"]),
    ("Ruy Lopez Marshall Attack", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "e8g8", "c2c3", "d7d5"]),
    ("Ruy Lopez Berlin Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4"]),
    ("Ruy Lopez Exchange", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6"]),

    # --- Sicilian Defense ---
    ("Sicilian Najdorf", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]),
    ("Sicilian Dragon", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]),
    ("Sicilian Scheveningen", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e6"]),
    ("Sicilian Classical", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6"]),
    ("Sicilian Sveshnikov", ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"]),
    ("Sicilian Accelerated Dragon", ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g7g6"]),
    ("Sicilian Alapin", ["e2e4", "c7c5", "c2c3", "d7d5", "e4d5", "d8d5", "d2d4"]),
    ("Sicilian Closed", ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2", "f8g7"]),
    ("Sicilian Rossolimo", ["e2e4", "c7c5", "g1f3", "b8c6", "f1b5"]),
    ("Sicilian Smith-Morra Gambit", ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3", "d4c3", "b1c3"]),

    # --- French Defense ---
    ("French Winawer", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"]),
    ("French Classical", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"]),
    ("French Tarrasch", ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5"]),
    ("French Advance", ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3"]),
    ("French Exchange", ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5", "e6d5"]),

    # --- Caro-Kann Defense ---
    ("Caro-Kann Main", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"]),
    ("Caro-Kann Advance", ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5"]),
    ("Caro-Kann Classical", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5", "e4g3", "f5g6"]),
    ("Caro-Kann Exchange", ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"]),

    # --- Scandinavian ---
    ("Scandinavian Main", ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]),
    ("Scandinavian Modern", ["e2e4", "d7d5", "e4d5", "g8f6"]),

    # --- Pirc / Modern ---
    ("Pirc Defense", ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4", "f8g7"]),
    ("Modern Defense", ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6"]),

    # --- Alekhine Defense ---
    ("Alekhine Defense", ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"]),

    # --- King's Gambit ---
    ("King's Gambit Accepted", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5"]),
    ("King's Gambit Declined", ["e2e4", "e7e5", "f2f4", "f8c5"]),

    # --- Scotch Game ---
    ("Scotch Game", ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"]),
    ("Scotch Four Knights", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "d2d4"]),

    # --- Petrov Defense ---
    ("Petrov Defense", ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3", "f6e4"]),

    # --- Vienna Game ---
    ("Vienna Game", ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4"]),

    # --- Philidor Defense ---
    ("Philidor Defense", ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "g8f6"]),

    # ===================== QUEEN'S PAWN (1.d4) =====================
    
    # --- Queen's Gambit ---
    ("Queen's Gambit Declined", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]),
    ("Queen's Gambit Accepted", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"]),
    ("Slav Defense", ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"]),
    ("Semi-Slav", ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6"]),
    ("Semi-Slav Meran", ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6", "e2e3", "b8d7", "f1d3", "d5c4", "d3c4", "b7b5"]),
    ("Tarrasch Defense", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5"]),
    ("QGD Exchange", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c4d5", "e6d5"]),

    # --- Indian Defenses ---
    ("King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]),
    ("King's Indian Classical", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5"]),
    ("King's Indian Samisch", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f3"]),
    ("Nimzo-Indian", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]),
    ("Nimzo-Indian Classical", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"]),
    ("Nimzo-Indian Rubinstein", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"]),
    ("Queen's Indian", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]),
    ("Bogo-Indian", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"]),
    ("Grunfeld Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]),
    ("Grunfeld Exchange", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3", "b2c3", "f8g7"]),
    ("Benoni Defense", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"]),
    ("Dutch Defense", ["d2d4", "f7f5"]),
    ("Dutch Stonewall", ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "e7e6", "f1g2", "d7d5"]),
    ("Dutch Leningrad", ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "g7g6", "f1g2", "f8g7"]),
    ("Catalan Opening", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2"]),

    # --- London System ---
    ("London System", ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"]),
    ("London vs KID", ["d2d4", "g8f6", "c1f4", "g7g6", "e2e3", "f8g7", "g1f3"]),

    # ===================== FLANK OPENINGS =====================
    
    # --- English Opening ---
    ("English Opening", ["c2c4", "e7e5"]),
    ("English Symmetrical", ["c2c4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6"]),
    ("English Reversed Sicilian", ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3"]),
    ("English Four Knights", ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"]),

    # --- Reti Opening ---
    ("Reti Opening", ["g1f3", "d7d5", "c2c4"]),
    ("Reti KIA", ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6"]),

    # --- Bird's Opening ---
    ("Bird's Opening", ["f2f4", "d7d5"]),

    # ===================== COMMON RESPONSES =====================
    
    # --- e4 e5 misc ---
    ("Four Knights Game", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]),
    ("Three Knights Game", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3"]),
    ("Center Game", ["e2e4", "e7e5", "d2d4", "e5d4", "d1d4"]),
    ("Bishop's Opening", ["e2e4", "e7e5", "f1c4"]),
    
    # --- d4 misc ---
    ("Trompowsky Attack", ["d2d4", "g8f6", "c1g5"]),
    ("Torre Attack", ["d2d4", "g8f6", "g1f3", "e7e6", "c1g5"]),
    ("Colle System", ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3", "e7e6", "f1d3"]),
    ("Veresov Opening", ["d2d4", "d7d5", "b1c3", "g8f6", "c1g5"]),
]


def generate_opening_positions(opening_book: list) -> tuple:
    """Walk through each opening line and extract every position + theory move."""
    fens = []
    best_moves = []
    evals_list = []
    opening_names = []
    seen = set()

    for name, moves_uci in opening_book:
        board = chess.Board()
        for i, uci in enumerate(moves_uci):
            fen = board.fen()
            # Avoid duplicates (transpositions)
            key = (fen, uci)
            if key not in seen:
                seen.add(key)
                fens.append(fen)
                best_moves.append(uci)
                # Eval: opening positions are roughly equal (near 0)
                # Slight positive for white (first-move advantage)
                eval_cp = 30.0 if board.turn == chess.WHITE else -15.0
                evals_list.append(eval_cp)
                opening_names.append(name)

            try:
                board.push_uci(uci)
            except ValueError:
                print(f"  WARNING: Invalid move {uci} in '{name}' at ply {i}")
                break

    print(f"Generated {len(fens):,} unique opening positions from {len(opening_book)} openings")
    return fens, best_moves, evals_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate opening book training data"
    )
    parser.add_argument("--out", type=str, default="data/train_openings.npz")
    args = parser.parse_args()

    t0 = time.time()
    fens, best_moves, evals = generate_opening_positions(OPENING_BOOK)

    if not fens:
        print("ERROR: No positions generated!")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        fens=np.array(fens, dtype=object),
        evals=np.array(evals, dtype=np.float32),
        best_moves=np.array(best_moves, dtype=object),
    )
    elapsed = time.time() - t0
    print(f"Saved {len(fens):,} opening positions to {args.out} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

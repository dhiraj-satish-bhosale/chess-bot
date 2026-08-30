"""
generate_data_pgn.py
-------------------------
Downloads and processes massive PGN databases (e.g. Lichess Elite or standard rated)
to create a high-quality supervised training dataset for the chess bot.

We extract the moves played by high-rated players (>2200 Elo) as the policy target,
and use the final game result as a proxy for evaluation (+300 cp for win, -300 for loss, 0 for draw).

Usage:
    python generate_data_pgn.py --url https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst --max-games 100000 --min-elo 2200
    python generate_data_pgn.py --file data/lichess_elite.pgn --max-positions 5000000 --out data/train_elite.npz
"""

import argparse
import io
import os
import time
import urllib.request
import zstandard as zstd
import numpy as np
import chess.pgn

def process_pgn_stream(file_obj, args):
    """Streams games from a PGN file object and extracts high-quality positions."""
    fens = []
    evals = []
    best_moves = []
    
    games_parsed = 0
    games_accepted = 0
    start_time = time.time()
    
    try:
        while True:
            if args.max_positions and len(fens) >= args.max_positions:
                break
            if args.max_games and games_accepted >= args.max_games:
                break

            game = chess.pgn.read_game(file_obj)
            if game is None:
                break  # EOF

            games_parsed += 1
            if games_parsed % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Parsed {games_parsed:,} games. Accepted {games_accepted:,}. Extracted {len(fens):,} positions. (Elapsed: {elapsed:.1f}s)")

            # Extract Headers
            headers = game.headers
            
            # Require standard chess
            if headers.get("Variant", "Standard") != "Standard":
                continue
                
            # Filter by Elo
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue
                
            if white_elo < args.min_elo or black_elo < args.min_elo:
                continue
                
            # Filter Time Controls (avoid bullet)
            tc = headers.get("TimeControl", "")
            if tc == "" or tc == "-":
                continue
            try:
                base_time = int(tc.split("+")[0])
                if base_time < 180:  # Skip games < 3 minutes
                    continue
            except ValueError:
                pass
                
            # Parse result for value target
            result = headers.get("Result", "*")
            if result == "1-0":
                # From White's perspective, this is +300 cp (winning)
                w_eval = 300.0
                b_eval = -300.0
            elif result == "0-1":
                w_eval = -300.0
                b_eval = 300.0
            elif result == "1/2-1/2":
                w_eval = 0.0
                b_eval = 0.0
            else:
                continue  # Abandoned game

            # Iterate through the game and extract positions
            board = game.board()
            positions_extracted = 0
            
            for move in game.mainline_moves():
                # Extract the state *before* the move is made
                fens.append(board.fen())
                best_moves.append(move.uci())
                
                # Assign the evaluation target relative to the current player
                if board.turn == chess.WHITE:
                    evals.append(w_eval)
                else:
                    evals.append(b_eval)
                
                board.push(move)
                positions_extracted += 1
                
                if args.max_positions and len(fens) >= args.max_positions:
                    break
                    
            if positions_extracted > 0:
                games_accepted += 1
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
        
    return fens, evals, best_moves

def main():
    parser = argparse.ArgumentParser(description="Generate supervised training data from PGNs")
    parser.add_argument("--url", type=str, default=None, help="URL to .pgn.zst file")
    parser.add_argument("--file", type=str, default=None, help="Local .pgn or .pgn.zst file")
    parser.add_argument("--out", type=str, default="data/train_elite.npz", help="Output .npz file")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum Elo of BOTH players to accept the game")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to accept")
    parser.add_argument("--max-positions", type=int, default=1000000, help="Maximum number of positions to extract")
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        print("Please provide either --url or --file")
        return
        
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # 1. Open the file or URL stream
    if args.url:
        import requests
        print(f"Streaming from URL: {args.url}")
        response = requests.get(args.url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        if args.url.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(response.raw)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='ignore')
        else:
            text_stream = io.TextIOWrapper(response.raw, encoding='utf-8', errors='ignore')
            
    else:
        print(f"Reading from file: {args.file}")
        if args.file.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            with open(args.file, "rb") as f:
                stream_reader = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='ignore')
                fens, evals, best_moves = process_pgn_stream(text_stream, args)
        else:
            with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
                fens, evals, best_moves = process_pgn_stream(f, args)
                
    if args.url: # If URL, we didn't process inside the with block above
        fens, evals, best_moves = process_pgn_stream(text_stream, args)
        text_stream.close()
        
    # 2. Save the dataset
    print(f"\nCompleted extraction.")
    print(f"Total positions: {len(fens):,}")
    print(f"Saving to {args.out}...")
    
    np.savez_compressed(
        args.out,
        fens=np.array(fens, dtype=str),
        evals=np.array(evals, dtype=np.float32),
        best_moves=np.array(best_moves, dtype=str)
    )
    print(f"Saved successfully. Size on disk: {os.path.getsize(args.out) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()

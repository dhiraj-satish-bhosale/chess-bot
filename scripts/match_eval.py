"""
scripts/match_eval.py
---------------------
Head-to-head evaluation match runner between:
1. Base Distilled Model (1900 Elo): alphazero_distilled.pt
2. RL-Trained Model: alphazero_best.pt (or any custom checkpoint)

Features:
- Alternates White and Black colors every game
- Configurable simulations and game count
- Live game-by-game reporting
- Exports PGN to review matches on Lichess / Chess.com
- Computes match score and win rates
"""

import os
import sys
import argparse
import datetime
import chess
import chess.pgn
import torch

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.mcts_bot import MCTSBot


def run_single_game(
    white_bot: MCTSBot,
    black_bot: MCTSBot,
    white_name: str,
    black_name: str,
    game_idx: int,
    simulations: int,
    opening_random_plies: int = 4,
    max_moves: int = 200,
) -> tuple[str, str, chess.pgn.Game]:
    """Plays a single game between two MCTSBot instances.
    
    Returns:
        (result_str, termination_reason, pgn_game)
        result_str is one of "1-0", "0-1", "1/2-1/2"
    """
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "AlphaZero RL vs Base 1900 Match"
    pgn_game.headers["Site"] = "Local Evaluation"
    pgn_game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_idx + 1)
    pgn_game.headers["White"] = white_name
    pgn_game.headers["Black"] = black_name

    node = pgn_game
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        current_bot = white_bot if board.turn == chess.WHITE else black_bot
        
        # In early opening (first N plies), use slight temperature to explore varied opening lines
        temp = 0.8 if move_count < opening_random_plies else 0.0

        move, visit_counts, root_val = current_bot.choose_move(
            board, simulations=simulations, temperature=temp
        )
        if move is None or move not in board.legal_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = legal_moves[0]

        san = board.san(move)
        node = node.add_variation(move)
        node.comment = f"eval={root_val:+.2f}"
        board.push(move)
        move_count += 1

    # Determine game outcome
    if board.is_checkmate():
        if board.turn == chess.BLACK:
            result = "1-0"
            reason = "Checkmate - White wins"
        else:
            result = "0-1"
            reason = "Checkmate - Black wins"
    elif board.is_stalemate():
        result = "1/2-1/2"
        reason = "Stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        reason = "Draw by insufficient material"
    elif board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = "1/2-1/2"
        reason = "Draw by repetition / 75-move rule"
    elif board.can_claim_threefold_repetition():
        result = "1/2-1/2"
        reason = "Draw by 3-fold repetition"
    elif board.can_claim_fifty_moves():
        result = "1/2-1/2"
        reason = "Draw by 50-move rule"
    else:
        # Reached max moves limit -> adjudicate based on material
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        w_mat = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        b_mat = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        if w_mat >= b_mat + 3:
            result = "1-0"
            reason = f"Adjudication (White +{w_mat - b_mat} material after {move_count} moves)"
        elif b_mat >= w_mat + 3:
            result = "0-1"
            reason = f"Adjudication (Black +{b_mat - w_mat} material after {move_count} moves)"
        else:
            result = "1/2-1/2"
            reason = f"Draw by move limit ({move_count} moves)"

    pgn_game.headers["Result"] = result
    pgn_game.headers["Termination"] = reason
    return result, reason, pgn_game


def main():
    parser = argparse.ArgumentParser(description="Head-to-head match between RL bot and Base 1900 bot.")
    parser.add_argument("--base-checkpoint", type=str, default="models/checkpoints/alphazero_distilled.pt",
                        help="Path to the base distilled model (1900 Elo)")
    parser.add_argument("--rl-checkpoint", type=str, default="models/checkpoints/alphazero_best.pt",
                        help="Path to the RL-trained model")
    parser.add_argument("--games", type=int, default=10, help="Total number of games to play")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--opening-plies", type=int, default=4,
                        help="Number of plies in the opening with soft exploration to vary opening lines")
    parser.add_argument("--pgn-out", type=str, default="models/checkpoints/h2h_base_vs_rl.pgn",
                        help="File to save all match PGNs")
    args = parser.parse_args()

    print("=" * 65)
    print("      ALPHAZERO HEAD-TO-HEAD MATCH: BASE (1900) vs RL BOT      ")
    print("=" * 65)
    print(f"Base 1900 Model : {args.base_checkpoint}")
    print(f"RL Trained Model: {args.rl_checkpoint}")
    print(f"Games to play   : {args.games}")
    print(f"Simulations/move: {args.simulations}")
    print(f"Opening Plies   : {args.opening_plies}")
    print(f"PGN Output      : {args.pgn_out}")
    print("-" * 65)

    if not os.path.exists(args.base_checkpoint):
        print(f"Error: Base checkpoint not found at {args.base_checkpoint}")
        return
    if not os.path.exists(args.rl_checkpoint):
        print(f"Error: RL checkpoint not found at {args.rl_checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}\n")

    print("Loading models into memory...")
    base_bot = MCTSBot(args.base_checkpoint, simulations=args.simulations, device=device)
    rl_bot = MCTSBot(args.rl_checkpoint, simulations=args.simulations, device=device)
    print("Both models loaded successfully!\n")

    rl_score = 0.0
    base_score = 0.0
    rl_wins = 0
    base_wins = 0
    draws = 0

    all_pgns = []

    for g in range(args.games):
        # Alternate colors: Even games -> RL is White, Odd games -> Base is White
        if g % 2 == 0:
            white_bot, black_bot = rl_bot, base_bot
            white_name, black_name = "RL-Trained-Bot", "Base-1900-Bot"
            rl_is_white = True
        else:
            white_bot, black_bot = base_bot, rl_bot
            white_name, black_name = "Base-1900-Bot", "RL-Trained-Bot"
            rl_is_white = False

        print(f"Game {g + 1:2d}/{args.games}: [White] {white_name} vs [Black] {black_name} ...", end="", flush=True)

        start_time = datetime.datetime.now()
        result, reason, pgn_game = run_single_game(
            white_bot=white_bot,
            black_bot=black_bot,
            white_name=white_name,
            black_name=black_name,
            game_idx=g,
            simulations=args.simulations,
            opening_random_plies=args.opening_plies,
        )
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        all_pgns.append(pgn_game)

        # Update scores
        if result == "1-0":
            if rl_is_white:
                rl_wins += 1
                rl_score += 1.0
                winner_tag = "RL Bot WON"
            else:
                base_wins += 1
                base_score += 1.0
                winner_tag = "Base 1900 WON"
        elif result == "0-1":
            if not rl_is_white:
                rl_wins += 1
                rl_score += 1.0
                winner_tag = "RL Bot WON"
            else:
                base_wins += 1
                base_score += 1.0
                winner_tag = "Base 1900 WON"
        else:
            draws += 1
            rl_score += 0.5
            base_score += 0.5
            winner_tag = "DRAW"

        print(f" Result: {result:7s} ({winner_tag}) | {reason} [{elapsed:.1f}s]")
        print(f"         Live Score -> RL Bot: {rl_score:.1f} | Base 1900: {base_score:.1f}")

    # Write all games to PGN file
    os.makedirs(os.path.dirname(os.path.abspath(args.pgn_out)), exist_ok=True)
    with open(args.pgn_out, "w", encoding="utf-8") as f:
        for pgn in all_pgns:
            print(pgn, file=f, end="\n\n")

    print("\n" + "=" * 65)
    print("                    FINAL MATCH SUMMARY                         ")
    print("=" * 65)
    print(f"Total Games Played  : {args.games}")
    print(f"RL-Trained Bot Wins : {rl_wins}  ({rl_wins / args.games * 100:.1f}%)")
    print(f"Base 1900 Bot Wins  : {base_wins}  ({base_wins / args.games * 100:.1f}%)")
    print(f"Draws               : {draws}  ({draws / args.games * 100:.1f}%)")
    print("-" * 65)
    print(f"Final Score: RL Bot [{rl_score:.1f}] - [{base_score:.1f}] Base 1900")
    rl_win_pct = (rl_score / args.games) * 100
    print(f"RL Bot Win Rate: {rl_win_pct:.1f}%")
    if rl_score > base_score:
        print("Verdict: >>> RL-Trained Bot is STRONGER than the 1900 base! <<<")
    elif rl_score < base_score:
        print("Verdict: >>> Base 1900 Bot outperformed the RL bot. <<<")
    else:
        print("Verdict: >>> Match ended in a DRAW (Equal Strength). <<<")
    print(f"Saved match PGN games to: {args.pgn_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()

"""
play.py
-------
CLI to play against the bot, watch it self-play, or evaluate a position.

Examples:
    # Play interactively against AlphaZero MCTS bot
    python play.py --checkpoint models/checkpoints/alphazero_best.pt \
                   --human_color white --simulations 800

    # Watch MCTS self-play
    python play.py --checkpoint models/checkpoints/alphazero_best.pt \
                   --self_play --self_play_moves 60 --simulations 400

    # Evaluate a position
    python play.py --checkpoint models/checkpoints/alphazero_best.pt \
                   --eval_only --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
"""
import argparse
import chess


def print_board(board: chess.Board):
    print(board.unicode(borders=True, empty_square="."))
    print()


# ---------------------------------------------------------------------------
# MCTS AlphaZero mode
# ---------------------------------------------------------------------------

def _mcts_eval_only(bot, fen: str, simulations: int):
    board = chess.Board(fen)
    print_board(board)

    move, visit_counts, root_val = bot.choose_move(
        board, simulations=simulations, temperature=0.0
    )

    side = "White" if board.turn == chess.WHITE else "Black"
    win_pct = (root_val + 1) / 2 * 100

    print(f"Best move for {side}: {board.san(move)}")
    print(f"Root value (Q): {root_val:+.4f}")
    print(f"Win probability: {win_pct:.1f}%")
    print(f"Total visits: {sum(visit_counts.values())}")

    # Show top moves
    sorted_moves = sorted(visit_counts.items(), key=lambda x: -x[1])
    print(f"\nTop moves:")
    for m, visits in sorted_moves[:5]:
        print(f"  {board.san(m):8s}  visits={visits}")


def _mcts_self_play(bot, simulations: int, max_moves: int):
    board = chess.Board()

    for i in range(max_moves):
        if board.is_game_over():
            break

        move, visit_counts, root_val = bot.choose_move(
            board, simulations=simulations, temperature=0.1,
        )
        san = board.san(move)
        board.push(move)

        mover = "White" if not board.turn else "Black"
        win_pct = (root_val + 1) / 2 * 100
        top_visits = max(visit_counts.values()) if visit_counts else 0
        print(f"{i+1:3d}. {mover:5s} plays {san:8s} "
              f"(Q={root_val:+.3f}, win%={win_pct:.1f}%, "
              f"top_visits={top_visits})")

    print()
    print_board(board)
    print("Result:", board.result(), "| Reason:", _game_over_reason(board))


def _mcts_interactive(bot, human_color: str, simulations: int):
    board = chess.Board()
    human_is_white = human_color.lower().startswith("w")

    while not board.is_game_over():
        print_board(board)
        human_turn = (board.turn == chess.WHITE) == human_is_white

        if human_turn:
            move_str = input("Your move (SAN or UCI, e.g. 'Nf3' or 'g1f3'): ").strip()
            if move_str.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                return
            try:
                move = board.parse_san(move_str)
            except ValueError:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal or unparseable move, try again.")
                    continue
            board.push(move)
        else:
            print(f"Bot is thinking ({simulations} MCTS simulations)...")
            move, visit_counts, root_val = bot.choose_move(
                board, simulations=simulations, temperature=0.0,
            )

            san = board.san(move)
            win_pct = (root_val + 1) / 2 * 100

            board.push(move)

            print(f"Bot plays: {san}  (Q={root_val:+.3f}, win%={win_pct:.1f}%)")

            # Show top alternatives
            sorted_moves = sorted(visit_counts.items(), key=lambda x: -x[1])
            if len(sorted_moves) > 1:
                # Undo to show SANs correctly
                board.pop()
                print("  Alternatives:")
                for m, v in sorted_moves[1:4]:
                    print(f"    {board.san(m):8s}  visits={v}")
                board.push(move)

    print_board(board)
    print("Game over:", board.result(), "-", _game_over_reason(board))


def _game_over_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.can_claim_fifty_moves():
        return "fifty-move rule"
    if board.can_claim_threefold_repetition():
        return "threefold repetition"
    return "ongoing / other"


def main():
    parser = argparse.ArgumentParser(
        description="Play chess against the AlphaZero bot"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/alphazero_distilled.pt")
    parser.add_argument("--human_color", type=str, default="white",
                        choices=["white", "black"])
    parser.add_argument("--fen", type=str, default=None)

    # MCTS options
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (AlphaZero mode)")

    # Mode
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--self_play", action="store_true")
    parser.add_argument("--self_play_moves", type=int, default=60)

    args = parser.parse_args()

    # AlphaZero MCTS mode
    from engine.mcts_bot import MCTSBot
    bot = MCTSBot(args.checkpoint, simulations=args.simulations)
    print(f"[AlphaZero mode] MCTS search, simulations={args.simulations}")

    if args.eval_only:
        fen = args.fen or chess.STARTING_FEN
        _mcts_eval_only(bot, fen, args.simulations)
    elif args.self_play:
        _mcts_self_play(bot, args.simulations, args.self_play_moves)
    else:
        _mcts_interactive(bot, args.human_color, args.simulations)


if __name__ == "__main__":
    main()

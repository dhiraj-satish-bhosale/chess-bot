"""
uci_engine.py
-------------
Minimal UCI (Universal Chess Interface) wrapper for the AlphaZero MCTS engine.

UCI is a simple line-based text protocol over stdin/stdout:
    GUI  -> "uci"                       engine identifies itself
    GUI  -> "isready"                   engine responds "readyok"
    GUI  -> "position startpos moves e2e4 e7e5 ..."   sets current position
    GUI  -> "go ..."                    engine searches, replies "bestmove e2e4"
    GUI  -> "quit"                      engine exits

Configurable UCI options:
    Simulations  — MCTS simulations per move (default 800)

Run standalone to test the handshake:
    echo "uci" | python uci_engine.py
"""
import os
import sys

# When bundled into a standalone .exe with PyInstaller, __file__ points
# into a temporary extraction folder, NOT to where the .exe actually
# lives -- so we must use sys.executable's location instead in that case.
# Also force stdout to be unbuffered/line-buffered: this matters once the
# script is no longer launched with 'python -u' (e.g. as a compiled exe
# or from inside a GUI), since GUIs like En Croissant read stdout line by
# line and will hang forever waiting for output that's stuck in a buffer.
if getattr(sys, "frozen", False):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # older Python without reconfigure(); explicit flush=True below covers it

import chess
import chess.polyglot
import chess.syzygy

# Default checkpoint path
ALPHAZERO_CHECKPOINT = os.path.join(SCRIPT_DIR, "models", "checkpoints", "alphazero_distilled.pt")

DEFAULT_SIMULATIONS = 800


DEFAULT_BOOK_PATH = os.path.join(SCRIPT_DIR, "data", "opening_book.bin")
DEFAULT_SYZYGY_PATH = os.path.join(SCRIPT_DIR, "data", "syzygy")


def log(msg):
    # UCI engines must never print anything to stdout except protocol
    # replies -- send debug info to stderr instead.
    print(msg, file=sys.stderr, flush=True)


def main():
    bot = None
    board = chess.Board()
    simulations = DEFAULT_SIMULATIONS
    book_path = DEFAULT_BOOK_PATH
    syzygy_path = DEFAULT_SYZYGY_PATH

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name AlphaZeroChessBot")
            print("id author Niraj")
            print(f"option name Simulations type spin default {DEFAULT_SIMULATIONS} min 50 max 10000")
            print(f"option name BookFile type string default {DEFAULT_BOOK_PATH}")
            print(f"option name SyzygyPath type string default {DEFAULT_SYZYGY_PATH}")
            print("uciok")
            sys.stdout.flush()

        elif line.startswith("setoption"):
            # e.g. "setoption name Simulations value 400"
            parts = line.split()
            if "Simulations" in parts:
                try:
                    idx = parts.index("value")
                    simulations = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            elif "BookFile" in parts:
                try:
                    idx = parts.index("value")
                    book_path = " ".join(parts[idx + 1:])
                except (ValueError, IndexError):
                    pass
            elif "SyzygyPath" in parts:
                try:
                    idx = parts.index("value")
                    syzygy_path = " ".join(parts[idx + 1:])
                except (ValueError, IndexError):
                    pass

        elif line == "isready":
            bot = _init_bot(simulations, syzygy_path)
            print("readyok")
            sys.stdout.flush()

        elif line == "ucinewgame":
            board = chess.Board()

        elif line.startswith("position"):
            board = _parse_position(line)

        elif line.startswith("go"):
            if bot is None:
                bot = _init_bot(simulations, syzygy_path)

            # 1. Check PolyGlot Opening Book for instant move (0.0s)
            book_move = None
            if os.path.exists(book_path):
                try:
                    with chess.polyglot.open_reader(book_path) as reader:
                        entry = reader.weighted_choice(board)
                        book_move = entry.move
                except Exception:
                    pass

            if book_move is not None:
                log(f"[book] instantaneous move {book_move.uci()} from opening book")
                print(f"bestmove {book_move.uci()}")
                sys.stdout.flush()
                continue

            time_budget = _compute_time_budget(line, board.turn)

            # Convert time budget to simulation count
            # Rough heuristic: ~400 sims/second on CPU, ~2000 on GPU
            sims = max(100, min(simulations, int(time_budget * 500)))

            move, root_val, sims_done = bot.choose_move_timed(
                board, time_budget=time_budget,
                max_simulations=max(sims, simulations),
            )
            log(f"[alphazero] {sims_done} sims, Q={root_val:+.3f}, "
                f"budget={time_budget:.2f}s")

            if move is None:
                print("bestmove 0000")
            else:
                print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif line == "stop":
            pass  # search completes within time budget

        elif line in ("quit",):
            break

        elif line == "ponderhit":
            pass  # pondering not implemented


_cached_bot = None


def _init_bot(simulations: int, syzygy_path: str = None):
    global _cached_bot

    if _cached_bot is not None:
        return _cached_bot

    checkpoint = ALPHAZERO_CHECKPOINT
    if not os.path.exists(checkpoint):
        log(f"AlphaZero checkpoint not found at {checkpoint}")
        return None

    from engine.mcts_bot import MCTSBot
    log(f"Loading AlphaZero model from {checkpoint} ...")
    bot = MCTSBot(checkpoint, simulations=simulations, tablebase_path=syzygy_path)
    # Warmup
    log("Warming up model...")
    bot.choose_move_timed(chess.Board(), time_budget=0.1, max_simulations=10)
    log("AlphaZero model loaded and warmed up.")

    _cached_bot = bot
    return bot


def _compute_time_budget(go_line: str, white_to_move: bool) -> float:
    """Parses a UCI 'go' command's time-control fields and returns how many
    seconds this move should get.

    Supports:
      go movetime <ms>              -- use exactly this many ms
      go wtime <ms> btime <ms> [winc <ms> binc <ms>]   -- standard clock
      go infinite / go (no time fields)                -- fall back to a
                                                            fixed default
                                                            budget

    Time-control formula: budget = remaining/30 + 80% of increment, then
    clamped to [MIN_BUDGET, MAX_BUDGET] so we never stall the GUI on a
    near-instant bullet clock nor blow the whole budget on one move.
    """
    MIN_BUDGET = 0.15
    MAX_BUDGET = 15.0
    DEFAULT_NO_CLOCK_BUDGET = 5.0

    tokens = go_line.split()

    def get_int(key):
        if key in tokens:
            idx = tokens.index(key)
            if idx + 1 < len(tokens):
                try:
                    return int(tokens[idx + 1])
                except ValueError:
                    return None
        return None

    movetime_ms = get_int("movetime")
    if movetime_ms is not None:
        return max(MIN_BUDGET, min(MAX_BUDGET, movetime_ms / 1000.0))

    wtime = get_int("wtime")
    btime = get_int("btime")
    winc = get_int("winc") or 0
    binc = get_int("binc") or 0

    remaining_ms = wtime if white_to_move else btime
    inc_ms = winc if white_to_move else binc

    if remaining_ms is None:
        return DEFAULT_NO_CLOCK_BUDGET  # "go infinite" or no time info given

    budget_sec = (remaining_ms / 30.0 + inc_ms * 0.8) / 1000.0
    return max(MIN_BUDGET, min(MAX_BUDGET, budget_sec))


def _parse_position(line: str) -> chess.Board:
    """Parses a UCI 'position' command:
       position startpos [moves e2e4 e7e5 ...]
       position fen <fen string> [moves ...]
    """
    tokens = line.split()
    board = chess.Board()

    if "startpos" in tokens:
        moves_idx = tokens.index("startpos") + 1
    elif "fen" in tokens:
        fen_start = tokens.index("fen") + 1
        if "moves" in tokens:
            moves_token_idx = tokens.index("moves")
            fen_str = " ".join(tokens[fen_start:moves_token_idx])
            moves_idx = moves_token_idx + 1
        else:
            fen_str = " ".join(tokens[fen_start:])
            moves_idx = len(tokens)
        board = chess.Board(fen_str)
    else:
        moves_idx = len(tokens)

    if "moves" in tokens:
        moves_token_idx = tokens.index("moves")
        for uci_move in tokens[moves_token_idx + 1:]:
            board.push_uci(uci_move)

    return board


if __name__ == "__main__":
    main()

"""
move_encoding.py
-----------------
AlphaZero-style 4672-dimensional move encoding.

Every legal chess move maps to a unique index in [0, 4672).  The encoding
uses 73 planes of 8×8, where each plane represents a move *type* (direction
+ distance for sliding pieces, or a specific knight jump, or an
underpromotion direction+piece) and the (row, col) within the plane
represents the *from-square* of the move.

Plane layout (73 planes total):
  Planes  0-55  (56): Queen-like moves — 8 directions × 7 distances
  Planes 56-63  ( 8): Knight moves    — 8 L-shaped jumps
  Planes 64-72  ( 9): Underpromotions — 3 directions × 3 piece types
                       (queen promotions are already covered by queen-like)

Moves are always encoded from the *current player's perspective*: for
Black, the board is flipped vertically so row 0 is always "our" back rank.
This matches AlphaZero's convention and means the policy head only ever
needs to learn one "colour" of patterns.
"""
import numpy as np
import chess

NUM_MOVE_PLANES = 73
TOTAL_MOVES = NUM_MOVE_PLANES * 8 * 8  # 4672

# ---------------------------------------------------------------------------
# Direction tables
# ---------------------------------------------------------------------------
# 8 queen-like directions as (delta_row, delta_col) in our coordinate
# system where row increases "forward" (toward the opponent).  For White
# that means rank increases; for Black (after flipping) it's the same.
QUEEN_DIRS = [
    (-1,  0),  # N  (forward)
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S  (backward)
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
]

# 8 knight jumps
KNIGHT_MOVES = [
    (-2, -1), (-2,  1),
    (-1, -2), (-1,  2),
    ( 1, -2), ( 1,  2),
    ( 2, -1), ( 2,  1),
]

# Underpromotion directions (relative to the pawn's square, from the
# current player's perspective — always moving "forward" = row decreasing).
# Column deltas: -1 (left capture), 0 (straight push), +1 (right capture).
UNDERPROMO_DIRS = [-1, 0, 1]

# Underpromotion pieces (queen is handled by normal queen-like planes).
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _flip_square(sq: int) -> int:
    """Flip a square vertically (mirror ranks): a1↔a8, b2↔b7, etc."""
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    return chess.square(file, 7 - rank)


def _square_to_rowcol(sq: int) -> tuple:
    """Convert a chess square to (row, col) in our board tensor layout.
    row 0 = rank 8 (top), row 7 = rank 1 (bottom). col 0 = file a."""
    return (7 - chess.square_rank(sq), chess.square_file(sq))


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a chess.Move to a flat index in [0, 4672).

    The move is encoded from the perspective of the side to move: for
    Black, both from-square and to-square are flipped vertically before
    computing direction/distance.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Flip for Black's perspective
    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    from_row, from_col = _square_to_rowcol(from_sq)
    to_row, to_col = _square_to_rowcol(to_sq)

    dr = to_row - from_row
    dc = to_col - from_col

    # --- Underpromotions (not queen) ---
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = UNDERPROMO_DIRS.index(dc)  # dc ∈ {-1, 0, 1}
        plane = 64 + dir_idx * 3 + piece_idx
        return plane * 64 + from_row * 8 + from_col

    # --- Knight moves ---
    if (dr, dc) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((dr, dc))
        plane = 56 + knight_idx
        return plane * 64 + from_row * 8 + from_col

    # --- Queen-like moves (includes queen promotions) ---
    # Determine direction and distance
    if dr == 0 and dc == 0:
        raise ValueError(f"Null move: {move}")

    # Normalize to unit direction
    if dr != 0:
        udr = dr // abs(dr)
    else:
        udr = 0
    if dc != 0:
        udc = dc // abs(dc)
    else:
        udc = 0

    direction = (udr, udc)
    distance = max(abs(dr), abs(dc))  # 1-7

    if direction not in QUEEN_DIRS:
        raise ValueError(f"Cannot encode move {move}: direction {direction} not queen-like or knight")

    dir_idx = QUEEN_DIRS.index(direction)
    plane = dir_idx * 7 + (distance - 1)  # planes 0-55
    return plane * 64 + from_row * 8 + from_col


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a flat index in [0, 4672) back to a chess.Move.

    Returns the move from the actual board perspective (un-flipped for Black).
    The returned move may not be legal — caller should check.
    """
    plane = index // 64
    remainder = index % 64
    from_row = remainder // 8
    from_col = remainder % 8

    if plane < 56:
        # Queen-like move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        udr, udc = QUEEN_DIRS[dir_idx]
        to_row = from_row + udr * distance
        to_col = from_col + udc * distance
        promotion = None
        # Check if this is a queen promotion (pawn reaching last rank)
        if to_row == 0:  # rank 8 from current player's perspective
            # Could be a pawn promotion — we'll set queen promotion and let
            # legality checking handle it
            from_sq_check = chess.square(from_col, 7 - from_row)
            if board.turn == chess.BLACK:
                from_sq_check = _flip_square(from_sq_check)
            piece = board.piece_at(from_sq_check)
            if piece is not None and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN
    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        dr, dc = KNIGHT_MOVES[knight_idx]
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = None
    else:
        # Underpromotion
        promo_plane = plane - 64
        dir_idx = promo_plane // 3
        piece_idx = promo_plane % 3
        dc = UNDERPROMO_DIRS[dir_idx]
        to_row = from_row - 1  # always one step forward
        to_col = from_col + dc
        promotion = UNDERPROMO_PIECES[piece_idx]

    # Bounds check
    if not (0 <= to_row < 8 and 0 <= to_col < 8):
        return None

    # Convert back to chess squares
    from_sq = chess.square(from_col, 7 - from_row)
    to_sq = chess.square(to_col, 7 - to_row)

    # Un-flip for Black
    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean mask of shape (4672,) — True for legal moves."""
    mask = np.zeros(TOTAL_MOVES, dtype=np.bool_)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            mask[idx] = True
        except (ValueError, IndexError):
            pass  # skip moves that can't be encoded (shouldn't happen)
    return mask


def get_legal_move_indices(board: chess.Board) -> dict:
    """Returns a dict mapping move_index -> chess.Move for all legal moves."""
    result = {}
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            result[idx] = move
        except (ValueError, IndexError):
            pass
    return result


def policy_to_move_probs(policy_logits: np.ndarray, board: chess.Board) -> list:
    """Given raw policy logits (4672,), returns a list of (move, probability)
    tuples for all legal moves, with probabilities summing to 1.

    Illegal moves are masked out before softmax.
    """
    legal_indices = get_legal_move_indices(board)
    if not legal_indices:
        return []

    indices = list(legal_indices.keys())
    logits = policy_logits[indices]

    # Numerically stable softmax over legal moves only
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    return [(legal_indices[idx], prob) for idx, prob in zip(indices, probs)]


if __name__ == "__main__":
    # Smoke test: encode and decode all legal moves from the starting position
    board = chess.Board()
    legal = list(board.legal_moves)
    print(f"Starting position: {len(legal)} legal moves")

    for move in legal:
        idx = move_to_index(move, board)
        decoded = index_to_move(idx, board)
        assert decoded == move, f"Roundtrip failed: {move} -> {idx} -> {decoded}"

    print("All starting-position moves roundtrip correctly.")

    # Test a position with promotions
    board2 = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    for move in board2.legal_moves:
        idx = move_to_index(move, board2)
        decoded = index_to_move(idx, board2)
        assert decoded == move, f"Roundtrip failed: {move} -> {idx} -> {decoded}"
    print("Promotion roundtrips correct.")

    # Test Black perspective
    board3 = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    for move in board3.legal_moves:
        idx = move_to_index(move, board3)
        decoded = index_to_move(idx, board3)
        assert decoded == move, f"Roundtrip failed (black): {move} -> {idx} -> {decoded}"
    print("Black perspective roundtrips correct.")

    mask = get_legal_move_mask(chess.Board())
    print(f"Legal move mask: {mask.sum()} legal moves out of {TOTAL_MOVES} slots")
    print("PASS")

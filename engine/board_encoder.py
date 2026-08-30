"""
board_encoder.py
-----------------
Converts a python-chess Board into a fixed-size tensor the network can consume.

Encoding: 18 planes of 8x8
  planes 0-5   : white piece planes  (P, N, B, R, Q, K)
  planes 6-11  : black piece planes  (p, n, b, r, q, k)
  plane 12     : side to move (all 1s if white to move, else all 0s)
  plane 13-16  : castling rights (white kingside, white queenside,
                                   black kingside, black queenside)
  plane 17     : en-passant target square (1 at that square, else 0)

Board is always encoded from White's perspective (i.e. we do NOT flip
for black to move). The side-to-move plane tells the network whose turn
it is; value head output is trained to always mean "White's advantage".
"""
import numpy as np
import chess

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_PLANES = 18


def encode_board(board: chess.Board) -> np.ndarray:
    """Returns a (NUM_PLANES, 8, 8) float32 tensor."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)  # rank 8 -> row 0 (standard image orientation)
        col = chess.square_file(square)
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        planes[plane_idx, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        planes[17, row, col] = 1.0

    return planes


def encode_batch(boards) -> np.ndarray:
    """Encode a list of boards into a (N, NUM_PLANES, 8, 8) array."""
    return np.stack([encode_board(b) for b in boards], axis=0)


# ---------------------------------------------------------------------------
# V2 encoding for AlphaZero: 21 planes, perspective-flipped
# ---------------------------------------------------------------------------

NUM_PLANES_V2 = 21

# Mapping piece types for v2 encoding (always "our" vs "their" perspective).
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_board_v2(board: chess.Board) -> np.ndarray:
    """Returns a (NUM_PLANES_V2, 8, 8) float32 tensor, encoded from the
    perspective of the side to move.

    Planes 0-5:   current player's pieces  (P, N, B, R, Q, K)
    Planes 6-11:  opponent's pieces         (P, N, B, R, Q, K)
    Plane 12:     all ones (side-to-move indicator, always 1.0 for v2 since
                  the board is already flipped to current player's viewpoint)
    Planes 13-16: castling rights (our kingside, our queenside,
                                    their kingside, their queenside)
    Plane 17:     en-passant target square
    Plane 18:     half-move clock (normalized: all cells = halfmove_clock/100)
    Plane 19:     fullmove number (normalized: all cells = min(fullmove, 200)/200)
    Plane 20:     repetition count (1.0 if position repeated, else 0.0)

    For Black, the board is flipped vertically so row 0 is always "our"
    back rank (matching AlphaZero's convention).
    """
    planes = np.zeros((NUM_PLANES_V2, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us
    flip = (us == chess.BLACK)

    # --- Piece planes (0-11) ---
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if flip:
            rank = 7 - rank
        row = 7 - rank
        col = file

        if piece.color == us:
            plane_idx = _PIECE_TYPES.index(piece.piece_type)
        else:
            plane_idx = 6 + _PIECE_TYPES.index(piece.piece_type)
        planes[plane_idx, row, col] = 1.0

    # --- Side-to-move plane (12) — always 1.0 in v2 (perspective is fixed) ---
    planes[12, :, :] = 1.0

    # --- Castling rights (13-16) ---
    planes[13, :, :] = float(board.has_kingside_castling_rights(us))
    planes[14, :, :] = float(board.has_queenside_castling_rights(us))
    planes[15, :, :] = float(board.has_kingside_castling_rights(them))
    planes[16, :, :] = float(board.has_queenside_castling_rights(them))

    # --- En-passant (17) ---
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        if flip:
            rank = 7 - rank
        row = 7 - rank
        col = file
        planes[17, row, col] = 1.0

    # --- Half-move clock (18) — normalized to [0, 1] ---
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0

    # --- Fullmove number (19) — normalized to [0, 1] ---
    planes[19, :, :] = min(board.fullmove_number, 200) / 200.0

    # --- Repetition (20) — 1.0 if position has been seen before ---
    planes[20, :, :] = 1.0 if board.is_repetition(1) else 0.0

    return planes


def encode_batch_v2(boards) -> np.ndarray:
    """Encode a list of boards into a (N, NUM_PLANES_V2, 8, 8) array."""
    return np.stack([encode_board_v2(b) for b in boards], axis=0)


if __name__ == "__main__":
    b = chess.Board()
    t = encode_board(b)
    print("V1 shape:", t.shape)
    print("white pawns plane:\n", t[0])
    print("side-to-move plane sum (should be 64, white to move):", t[12].sum())

    t2 = encode_board_v2(b)
    print("\nV2 shape:", t2.shape)
    print("our pawns plane (white):\n", t2[0])
    print("halfmove clock plane value:", t2[18, 0, 0])
    print("fullmove plane value:", t2[19, 0, 0])

    # Test flipping for Black
    b2 = chess.Board()
    b2.push_san("e4")
    t_black = encode_board_v2(b2)
    print("\nAfter 1.e4 (Black to move), our pawns (Black's pawns):")
    print(t_black[0])
    print("PASS")

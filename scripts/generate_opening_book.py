"""
scripts/generate_opening_book.py
--------------------------------
Generates a PolyGlot opening book (data/opening_book.bin) from standard 
Grandmaster opening theory lines and master variations.
"""

import os
import struct
import chess
import chess.pgn
import chess.polyglot
import io

MASTER_OPENINGS_PGN = """
[Event "Ruy Lopez: Berlin"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. d4 Nd6 6. Bxc6 dxc6 7. dxe5 Nf5 8. Qxd8+ Kxd8 *

[Event "Ruy Lopez: Closed / Morphy"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 *

[Event "Ruy Lopez: Open"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Nxe4 6. d4 b5 7. Bb3 d5 8. dxe5 Be6 9. c3 Be7 10. Nbd2 *

[Event "Italian Game: Giuoco Piano"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7 9. h3 h6 *

[Event "Italian Game: Two Knights"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Be7 5. O-O O-O 6. Re1 d6 7. c3 Na5 8. Bb5 a6 9. Ba4 b5 *

[Event "Sicilian: Najdorf"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 *

[Event "Sicilian: Classical / Richter-Rauzer"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 Nc6 6. Bg5 e6 7. Qd2 a6 8. O-O-O Bd7 9. f4 Be7 *

[Event "Sicilian: Dragon"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 6. Be3 Bg7 7. f3 O-O 8. Qd2 Nc6 9. Bc4 Bd7 10. O-O-O *

[Event "Sicilian: Scheveningen / Taimanov"]
1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nc6 5. Nc3 Qc7 6. Be3 a6 7. Qd2 Nf6 8. O-O-O Be7 9. f3 O-O *

[Event "Sicilian: Kan"]
1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 a6 5. Bd3 Nf6 6. O-O d6 7. c4 g6 8. Nc3 Bg7 9. Be3 O-O *

[Event "French: Winawer"]
1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O 8. Bd3 Nbc6 9. Qh5 Ng6 *

[Event "French: Classical"]
1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. e5 Nfd7 5. f4 c5 6. Nf3 Nc6 7. Be3 a6 8. Qd2 b5 9. Be2 Be7 *

[Event "French: Tarrasch"]
1. e4 e6 2. d4 d5 3. Nd2 c5 4. Ngf3 Nf6 5. exd5 exd5 6. Bb5+ Bd7 7. Bxd7+ Nbxd7 8. O-O Be7 9. dxc5 Nxc5 *

[Event "Caro-Kann: Classical"]
1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 8. h5 Bh7 9. Bd3 Bxd3 10. Qxd3 e6 *

[Event "Caro-Kann: Advance"]
1. e4 c6 2. d4 d5 3. e5 Bf5 4. Nf3 e6 5. Be2 c5 6. Be3 Nc6 7. O-O cxd4 8. Nxd4 Nxd4 9. Bxd4 Ne7 *

[Event "Queen's Gambit Declined: Tartakower"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 h6 6. Bh4 O-O 7. Nf3 b6 8. cxd5 Nxd5 9. Bxe7 Qxe7 10. Nxd5 exd5 *

[Event "Queen's Gambit Declined: Semi-Slav"]
1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. e3 Nbd7 6. Qc2 Bd6 7. Bd3 O-O 8. O-O dxc4 9. Bxc4 b5 10. Be2 Bb7 *

[Event "Queen's Gambit Accepted"]
1. d4 d5 2. c4 dxc4 3. Nf3 Nf6 4. e3 e6 5. Bxc4 c5 6. O-O a6 7. Qe2 b5 8. Bb3 Bb7 9. Rd1 Nbd7 *

[Event "Slav Defense"]
1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5 6. e3 e6 7. Bxc4 Bb4 8. O-O O-O 9. Qe2 Nbd7 *

[Event "King's Indian: Mar del Plata"]
1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Be3 f5 *

[Event "Nimzo-Indian: Rubinstein"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 *

[Event "Grünfeld Defense: Exchange"]
1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7 7. Nf3 c5 8. Rb1 O-O 9. Be2 cxd4 10. cxd4 Qa5+ *

[Event "English Opening: Symmetrical"]
1. c4 c5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. Nf3 Nf6 6. O-O O-O 7. d4 cxd4 8. Nxd4 Nxd4 9. Qxd4 d6 *

[Event "English Opening: Four Knights"]
1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 Bb4 5. Bg2 O-O 6. O-O e4 7. Ng5 Bxc3 8. bxc3 Re8 9. f3 exf3 *

[Event "Reti Opening"]
1. Nf3 d5 2. g3 Nf6 3. Bg2 c6 4. O-O Bf5 5. d3 e6 6. Nbd2 h6 7. Qe1 Be7 8. e4 Bh7 9. Qe2 O-O *
"""

def encode_move_polyglot(move: chess.Move) -> int:
    to_sq = move.to_square & 0x3F
    from_sq = (move.from_square & 0x3F) << 6
    prom_part = ((move.promotion - 1) & 0x7) << 12 if move.promotion else 0
    return to_sq | from_sq | prom_part

def create_polyglot_book(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    entries = []
    seen = set()
    
    pgn_io = io.StringIO(MASTER_OPENINGS_PGN)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        
        board = game.board()
        for node in game.mainline():
            move = node.move
            key = chess.polyglot.zobrist_hash(board)
            entry_key = (key, move.uci())
            if entry_key not in seen:
                seen.add(entry_key)
                weight = 100
                learn = 0
                entries.append((key, move, weight, learn))
            board.push(move)

    # Sort entries by Polyglot key as required by the binary specification
    entries.sort(key=lambda x: x[0])

    with open(output_path, "wb") as f:
        for key, move, weight, learn in entries:
            raw_move = encode_move_polyglot(move)
            f.write(struct.pack(">QHHI", key, raw_move, weight, learn))

    print(f"Successfully generated PolyGlot book with {len(entries)} master moves at: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "opening_book.bin")
    create_polyglot_book(out)

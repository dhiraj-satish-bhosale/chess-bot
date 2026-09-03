"""
mcts_bot.py
------------
High-level bot interface for playing chess with MCTS + AlphaZeroNet.

Provides the main API (choose_move, choose_move_timed) and uses
PUCT-based Monte Carlo Tree Search internally.
"""
import time
import chess
import torch

from engine.mcts import MCTS
from models.network import ChessValueNet, load_model


class MCTSBot:
    """High-level MCTS-based chess bot.

    Args:
        checkpoint_path: Path to an AlphaZeroNet checkpoint.
        simulations: Default number of MCTS simulations per move.
        c_puct: PUCT exploration constant.
        device: torch device ('cuda' / 'cpu' / None for auto).
    """

    def __init__(self, checkpoint_path: str, simulations: int = 800,
                 c_puct: float = 2.5, device=None, tablebase_path: str = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        )
        self.net = load_model(checkpoint_path, self.device, output_policy=True)
        self.simulations = simulations
        self.mcts = MCTS(
            self.net, device=self.device, c_puct=c_puct, tablebase_path=tablebase_path
        )
        # Tree reuse: keep the root from the last search so we can reuse
        # the subtree when the opponent makes the expected move.
        self._last_root = None
        self._last_board_fen = None

    def choose_move(self, board: chess.Board, simulations: int = None,
                    temperature: float = 0.0, add_noise: bool = False):
        """Select a move using MCTS.

        Args:
            board: Current position.
            simulations: Number of MCTS sims (overrides default).
            temperature: Move selection temperature (0 = greedy).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            (move, visit_counts_dict, root_value)
        """
        sims = simulations or self.simulations

        # Safe tree reuse logic: only reuse if node was expanded and has children
        root = None
        if self._last_root is not None and len(board.move_stack) > 0:
            last_move = board.move_stack[-1]
            temp = board.copy()
            temp.pop()
            if temp.fen() == self._last_board_fen and last_move in self._last_root.children:
                candidate = self._last_root.children[last_move]
                if candidate.is_expanded and len(candidate.children) > 0:
                    root = candidate
                    root.parent = None

        root = self.mcts.search(board, num_simulations=sims, root=root, add_noise=add_noise)

        # Extract info
        move = self.mcts.select_move(root, temperature=temperature)
        visit_counts = {
            m: child.visit_count for m, child in root.children.items()
        }
        root_value = root.q_value

        # Store for potential tree reuse
        if move and move in root.children:
            self._last_root = root.children[move]
            temp = board.copy()
            temp.push(move)
            self._last_board_fen = temp.fen()
        else:
            self._last_root = None
            self._last_board_fen = None

        return move, visit_counts, root_value

    def choose_move_timed(self, board: chess.Board, time_budget: float,
                          max_simulations: int = 3200,
                          temperature: float = 0.0, add_noise: bool = False):
        """Select a move with a time budget.

        Runs MCTS simulations until the time budget is exhausted or
        max_simulations is reached.

        Args:
            board: Current position.
            time_budget: Maximum seconds to spend.
            max_simulations: Cap on total simulations.
            temperature: Move selection temperature.
            add_noise: Whether to add Dirichlet noise.

        Returns:
            (move, root_value, simulations_done)
        """
        # Safe tree reuse logic: only reuse if node was expanded and has children
        root = None
        if self._last_root is not None and len(board.move_stack) > 0:
            last_move = board.move_stack[-1]
            temp = board.copy()
            temp.pop()
            if temp.fen() == self._last_board_fen and last_move in self._last_root.children:
                candidate = self._last_root.children[last_move]
                if candidate.is_expanded and len(candidate.children) > 0:
                    root = candidate
                    root.parent = None

        root, sims_done = self.mcts.search_timed(
            board, time_budget=time_budget, max_simulations=max_simulations,
            root=root, add_noise=add_noise
        )

        move = self.mcts.select_move(root, temperature=temperature)

        # Store for potential tree reuse
        if move and move in root.children:
            self._last_root = root.children[move]
            temp = board.copy()
            temp.push(move)
            self._last_board_fen = temp.fen()
        else:
            self._last_root = None
            self._last_board_fen = None

        return move, root.q_value, sims_done

    def get_principal_variation(self, root, board: chess.Board, max_depth: int = 10) -> list:
        """Extract the principal variation (PV) from the search tree.

        Returns a list of SAN move strings following the most-visited path.
        """
        pv = []
        node = root
        sim_board = board.copy()

        for _ in range(max_depth):
            if not node.children:
                break
            # Follow the most-visited child
            best_child = max(node.children.values(), key=lambda c: c.visit_count)
            pv.append(sim_board.san(best_child.move))
            sim_board.push(best_child.move)
            node = best_child

        return pv

    def get_move_stats(self, root, board: chess.Board, top_n: int = 5) -> list:
        """Get statistics for the top N moves.

        Returns a list of dicts with 'move_san', 'visits', 'q_value', 'prior'.
        """
        if not root.children:
            return []

        sorted_children = sorted(
            root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True,
        )

        stats = []
        for move, child in sorted_children[:top_n]:
            stats.append({
                "move_san": board.san(move),
                "move_uci": move.uci(),
                "visits": child.visit_count,
                "q_value": child.q_value,
                "prior": child.prior,
                "win_pct": (child.q_value + 1) / 2 * 100,  # convert to %
            })
        return stats


# Use the same MCTSNode class for timed search
from engine.mcts import MCTSNode as _MCTSNodeForTimed


if __name__ == "__main__":
    import sys

    # Quick test with a random network
    net = ChessValueNet(output_policy=True)
    net.eval()
    device = torch.device("cpu")
    net.to(device)

    # Save a temp checkpoint
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "test_az.pt")
    torch.save({
        "model_state_dict": net.state_dict(),
        "channels": 128,
        "num_res_blocks": 10,
    }, tmp)

    bot = MCTSBot(tmp, simulations=100, device=device)
    board = chess.Board()

    print("Testing choose_move (100 sims)...")
    move, visits, root_val = bot.choose_move(board, simulations=100)
    print(f"  Best move: {board.san(move)}, root Q: {root_val:+.3f}")
    print(f"  Top visits: {sorted(visits.values(), reverse=True)[:5]}")

    print("\nTesting choose_move_timed (1 second budget)...")
    move, val, sims = bot.choose_move_timed(board, time_budget=1.0)
    print(f"  Best move: {board.san(move)}, sims: {sims}, Q: {val:+.3f}")

    os.remove(tmp)
    print("PASS")

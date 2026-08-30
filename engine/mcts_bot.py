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
            "cuda" if torch.cuda.is_available() else "cpu"
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

        root = self.mcts.search(board, num_simulations=sims, add_noise=add_noise)

        # Extract info
        move = self.mcts.select_move(root, temperature=temperature)
        visit_counts = {
            m: child.visit_count for m, child in root.children.items()
        }
        root_value = root.q_value

        # Store for potential tree reuse
        self._last_root = root
        self._last_board_fen = board.fen()

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
        deadline = time.time() + max(0.05, time_budget)

        from engine.board_encoder import encode_board_v2
        from engine.move_encoding import get_legal_move_indices

        root = _MCTSNodeForTimed()

        # Evaluate and expand root
        policy_logits, root_value = self.mcts._evaluate(board)
        root.expand(policy_logits, board)

        if add_noise:
            root.add_dirichlet_noise()

        if not root.children:
            # No legal moves
            legal = list(board.legal_moves)
            return (legal[0] if legal else None), 0.0, 0

        sims_done = 0
        while sims_done < max_simulations and time.time() < deadline:
            # Run a batch of simulations
            batch_size = min(32, max_simulations - sims_done)
            for _ in range(batch_size):
                node = root
                sim_board = board.copy()

                # Selection
                while not node.is_leaf() and node.children:
                    node = node.select_child(self.mcts.c_puct)
                    sim_board.push(node.move)

                # Evaluation & Expansion
                if sim_board.is_game_over():
                    if sim_board.is_checkmate():
                        value = -1.0
                    else:
                        value = 0.0
                else:
                    policy_logits, value = self.mcts._evaluate(sim_board)
                    node.expand(policy_logits, sim_board)

                # Backpropagation
                node.backpropagate(value)
                sims_done += 1

            if time.time() >= deadline:
                break

        move = self.mcts.select_move(root, temperature=temperature)
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

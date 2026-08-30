"""
mcts.py
--------
PUCT-based Monte Carlo Tree Search following the AlphaZero paper.

This is the "brain" of the AlphaZero system: at each move, it builds a
search tree guided by the neural network's policy (which branches to
explore) and value (how good a leaf looks), selecting the move with the
most visits after a fixed number of simulations.

Key concepts:
  - Selection:  Walk down the tree choosing the child with the highest
                UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
  - Expansion:  At a leaf, query the neural network for (policy, value).
  - Backup:     Propagate the value back up the tree (negated at each level
                because the players alternate).
  - Root noise: Add Dirichlet noise to the root's prior probabilities to
                ensure exploration during self-play training.
"""
import os
import time
import math
import numpy as np
import chess
import torch

from engine.board_encoder import encode_board_v2
from engine.move_encoding import (
    move_to_index, index_to_move, get_legal_move_mask,
    get_legal_move_indices, TOTAL_MOVES,
)


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = [
        "parent", "move", "prior", "visit_count", "total_value",
        "children", "is_expanded", "board_hash", "virtual_loss"
    ]

    def __init__(self, parent=None, move=None, prior: float = 0.0):
        self.parent = parent
        self.move = move              # chess.Move that led to this node
        self.prior = prior            # P(s, a) from the policy network
        self.visit_count = 0          # N(s, a)
        self.total_value = 0.0        # W(s, a)
        self.children = {}            # {chess.Move: MCTSNode}
        self.is_expanded = False
        self.board_hash = None
        self.virtual_loss = 0

    @property
    def q_value(self) -> float:
        """Mean value Q(s,a) = W/N.  Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def add_virtual_loss(self, loss_val: int = 1):
        self.virtual_loss += loss_val

    def revert_virtual_loss(self, loss_val: int = 1):
        self.virtual_loss -= loss_val

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Select the child with the highest PUCT score.

        Q values are stored from the child's perspective (opponent), so we
        negate them: a child where the opponent is losing (Q < 0) is good
        for us (score contribution > 0).
        """
        sqrt_total = math.sqrt(max(1, self.visit_count))
        best_score = -float("inf")
        best_child = None

        for child in self.children.values():
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = -child.q_value + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_logits: np.ndarray, board: chess.Board):
        """Expand this leaf node using the neural network's policy output.

        Creates child nodes for each legal move with their prior probabilities.
        """
        self.is_expanded = True
        legal_indices = get_legal_move_indices(board)

        if not legal_indices:
            return  # terminal node (no legal moves)

        # Extract logits for legal moves and softmax them
        indices = list(legal_indices.keys())
        logits = policy_logits[indices].astype(np.float64)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        for i, (idx, move) in enumerate(zip(indices, [legal_indices[k] for k in indices])):
            child = MCTSNode(parent=self, move=move, prior=float(probs[i]))
            self.children[move] = child

    def backpropagate(self, value: float):
        """Update this node and all ancestors with the search result.

        Value is negated at each level because the players alternate.
        """
        node = self
        sign = 1.0
        while node is not None:
            node.visit_count += 1
            node.total_value += sign * value
            node = node.parent
            sign = -sign

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25):
        """Add Dirichlet noise to the root node's prior probabilities.

        This ensures that the MCTS explores broadly during self-play
        training, even in positions where the network is very confident.

        Formula: P'(s,a) = (1 - ε) * P(s,a) + ε * Dir(α)
        """
        if not self.children:
            return
        noise = np.random.dirichlet([alpha] * len(self.children))
        for i, child in enumerate(self.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


class MCTS:
    """Full PUCT-based Monte Carlo Tree Search engine.

    Uses a neural network (policy + value) to guide the search.
    """

    def __init__(self, net, device=None, c_puct: float = 2.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 tablebase_path: str = None):
        self.net = net
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.tt = {}  # Transposition Table cache: fen/key -> (policy_logits, value)
        self.tablebase = None
        if tablebase_path and os.path.exists(tablebase_path):
            try:
                import chess.syzygy
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
            except Exception as e:
                print(f"Warning: Failed to load Syzygy tablebase from {tablebase_path}: {e}")

    def clear_tt(self):
        """Clear the transposition table."""
        self.tt.clear()

    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """Run the neural network on a position (cached via Transposition Table).

        Returns:
            policy_logits: numpy array (4672,)
            value: float in (-1, 1) from current player's perspective
        """
        key = board._transposition_key() if hasattr(board, "_transposition_key") else board.fen()
        if key in self.tt:
            return self.tt[key]

        encoded = encode_board_v2(board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).float().to(self.device)
        policy_logits, value = self.net(tensor)
        res = (policy_logits.squeeze(0).cpu().numpy(), value.item())
        
        # Keep TT bounded
        if len(self.tt) < 250000:
            self.tt[key] = res
        return res

    @torch.no_grad()
    def _evaluate_batch(self, boards: list):
        """Run the neural network on a batch of positions.

        Returns:
            policy_logits_list: list of numpy arrays (4672,)
            values_list: list of floats
        """
        if not boards:
            return [], []
        encoded = np.stack([encode_board_v2(b) for b in boards], axis=0)
        tensor = torch.from_numpy(encoded).float().to(self.device)
        policy_logits, values = self.net(tensor)
        policy_logits = policy_logits.cpu().numpy()
        values = values.cpu().numpy()
        return [policy_logits[i] for i in range(len(boards))], \
               [float(values[i]) for i in range(len(boards))]

    def search(self, board: chess.Board, num_simulations: int,
               root: MCTSNode = None, add_noise: bool = True) -> MCTSNode:
        """Run MCTS for num_simulations iterations from the given position."""
        if root is None or not root.is_expanded or not root.children:
            root = MCTSNode()
            policy_logits, root_value = self._evaluate(board)
            root.expand(policy_logits, board)
            if add_noise:
                root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        if not root.children:
            return root  # no legal moves (game over at root)

        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()

            # --- Selection: walk down the tree ---
            while not node.is_leaf() and node.children:
                node = node.select_child(self.c_puct)
                sim_board.push(node.move)

            # --- Evaluation & Expansion ---
            if sim_board.is_game_over():
                # Terminal node: use true game result
                if sim_board.is_checkmate():
                    value = -1.0
                else:
                    value = 0.0
            elif self.tablebase and len(sim_board.piece_map()) <= 5:
                try:
                    wdl = self.tablebase.probe_wdl(sim_board)
                    value = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)
                except Exception:
                    policy_logits, value = self._evaluate(sim_board)
                    node.expand(policy_logits, sim_board)
            else:
                # Non-terminal leaf: expand and evaluate with the network
                policy_logits, value = self._evaluate(sim_board)
                node.expand(policy_logits, sim_board)

            # --- Backpropagation ---
            node.backpropagate(value)

        return root

    def search_timed(self, board: chess.Board, time_budget: float,
                     max_simulations: int = 3200, root: MCTSNode = None,
                     add_noise: bool = False) -> tuple:
        """Run MCTS within a time budget.
        
        Returns:
            (root_node, simulations_completed)
        """
        deadline = time.time() + max(0.05, time_budget)
        
        if root is None or not root.is_expanded or not root.children:
            root = MCTSNode()
            policy_logits, root_value = self._evaluate(board)
            root.expand(policy_logits, board)
            if add_noise:
                root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        if not root.children:
            return root, 0

        sims_done = 0
        while sims_done < max_simulations and time.time() < deadline:
            node = root
            sim_board = board.copy()

            # Selection
            while not node.is_leaf() and node.children:
                node = node.select_child(self.c_puct)
                sim_board.push(node.move)

            # Evaluation & Expansion
            if sim_board.is_game_over():
                if sim_board.is_checkmate():
                    value = -1.0
                else:
                    value = 0.0
            elif self.tablebase and len(sim_board.piece_map()) <= 5:
                try:
                    wdl = self.tablebase.probe_wdl(sim_board)
                    value = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)
                except Exception:
                    policy_logits, value = self._evaluate(sim_board)
                    node.expand(policy_logits, sim_board)
            else:
                policy_logits, value = self._evaluate(sim_board)
                node.expand(policy_logits, sim_board)

            # Backpropagation
            node.backpropagate(value)
            sims_done += 1

        return root, sims_done

    def search_batched(self, board: chess.Board, num_simulations: int,
                       batch_size: int = 8, add_noise: bool = True) -> MCTSNode:
        """Run MCTS with batched neural network evaluation."""
        root = MCTSNode()
        
        # Evaluate root and expand
        policy_logits, root_value = self._evaluate(board)
        root.expand(policy_logits, board)

        if add_noise:
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        if not root.children:
            return root  # no legal moves

        sims_done = 0
        while sims_done < num_simulations:
            current_batch_size = min(batch_size, num_simulations - sims_done)
            leaves = []
            boards = []
            
            # --- Selection (collecting a batch of leaves) ---
            for _ in range(current_batch_size):
                node = root
                sim_board = board.copy()
                path = []
                
                while not node.is_leaf() and node.children:
                    node = node.select_child(self.c_puct)
                    path.append(node)
                    sim_board.push(node.move)
                
                # Apply virtual loss to the path to discourage other threads/sims from picking it
                for n in path:
                    n.add_virtual_loss()
                
                leaves.append((node, path))
                boards.append(sim_board)
                
            # --- Evaluation & Expansion ---
            # Separate terminal and non-terminal states
            eval_boards = []
            eval_indices = []
            terminal_values = []
            
            for i, sim_board in enumerate(boards):
                node, _ = leaves[i]
                if sim_board.is_game_over():
                    val = -1.0 if sim_board.is_checkmate() else 0.0
                    terminal_values.append((i, val))
                elif self.tablebase and len(sim_board.piece_map()) <= 5:
                    try:
                        wdl = self.tablebase.probe_wdl(sim_board)
                        val = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)
                        terminal_values.append((i, val))
                    except Exception:
                        eval_boards.append(sim_board)
                        eval_indices.append(i)
                else:
                    eval_boards.append(sim_board)
                    eval_indices.append(i)
                    
            # Batch evaluate all non-terminal nodes
            policy_list, value_list = self._evaluate_batch(eval_boards)
            
            # --- Backpropagation ---
            for j, (i, val) in enumerate(terminal_values):
                node, path = leaves[i]
                for n in path:
                    n.revert_virtual_loss()
                node.backpropagate(val)
                sims_done += 1
                
            for j, i in enumerate(eval_indices):
                node, path = leaves[i]
                for n in path:
                    n.revert_virtual_loss()
                node.expand(policy_list[j], boards[i])
                node.backpropagate(value_list[j])
                sims_done += 1
                
        return root

    def get_policy(self, root: MCTSNode, temperature: float = 1.0) -> tuple:
        """Extract the MCTS policy from the root's visit counts.

        Args:
            root: The root MCTSNode after search.
            temperature: Controls exploration vs exploitation.
                         τ=1.0: proportional to visit counts (exploratory).
                         τ→0:   greedy (pick the most-visited move).

        Returns:
            moves: list of chess.Move
            probs: numpy array of probabilities (sums to 1)
        """
        if not root.children:
            return [], np.array([])

        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

        if temperature < 1e-8:
            # Greedy: pick the move with the most visits
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature-scaled softmax over visit counts
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return moves, probs

    def get_policy_target(self, root: MCTSNode, board: chess.Board) -> np.ndarray:
        """Convert the MCTS visit counts to a full 4672-dimensional policy
        target vector (for training the network).

        Returns a numpy array of shape (4672,) with visit-count probabilities.
        """
        target = np.zeros(TOTAL_MOVES, dtype=np.float32)
        if not root.children:
            return target

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return target

        for move, child in root.children.items():
            try:
                idx = move_to_index(move, board)
                target[idx] = child.visit_count / total_visits
            except (ValueError, IndexError):
                pass

        return target

    def select_move(self, root: MCTSNode, temperature: float = 1.0) -> chess.Move:
        """Select a move from the root using the MCTS policy.

        Args:
            root: Root node after search.
            temperature: Controls exploration (1.0) vs exploitation (→0).

        Returns:
            The selected chess.Move.
        """
        moves, probs = self.get_policy(root, temperature)
        if not moves:
            return None
        if temperature < 1e-8:
            return moves[np.argmax(probs)]
        return moves[np.random.choice(len(moves), p=probs)]


if __name__ == "__main__":
    # Smoke test with a randomly initialized network
    from models.network import ChessValueNet

    net = ChessValueNet(output_policy=True)
    net.eval()
    device = torch.device("cpu")
    net.to(device)

    mcts = MCTS(net, device=device, c_puct=2.5)

    board = chess.Board()
    print("Running MCTS with 100 simulations from start position...")
    root = mcts.search(board, num_simulations=100, add_noise=True)

    moves, probs = mcts.get_policy(root, temperature=1.0)
    print(f"\nTop moves (of {len(moves)} legal):")
    sorted_pairs = sorted(zip(moves, probs), key=lambda x: -x[1])
    for m, p in sorted_pairs[:5]:
        child = root.children[m]
        print(f"  {board.san(m):8s}  visits={child.visit_count:4d}  "
              f"prob={p:.3f}  Q={child.q_value:+.3f}")

    best = mcts.select_move(root, temperature=0.0)
    print(f"\nBest move (greedy): {board.san(best)}")

    target = mcts.get_policy_target(root, board)
    print(f"Policy target shape: {target.shape}, sum: {target.sum():.4f}")
    print("PASS")

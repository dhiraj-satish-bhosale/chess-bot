import os
import time
import numpy as np
import torch
import chess
import concurrent.futures

from engine.mcts import MCTS
from engine.board_encoder import encode_board_v2
from engine.move_encoding import move_to_index
from models.network import load_model

def play_one_game(
    checkpoint_path: str,
    simulations: int,
    temperature_moves: int,
    max_moves: int,
    resign_threshold: float,
    resign_count: int,
    device_str: str,
    c_puct: float,
    dirichlet_alpha: float
):
    device = torch.device(device_str)
    net = load_model(checkpoint_path, device, output_policy=True)
    net.eval()
    
    mcts = MCTS(net, device=device, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha)
    board = chess.Board()
    
    examples = []
    resignation_counter = 0
    
    for move_idx in range(max_moves):
        if board.is_game_over():
            break
            
        temp = 1.0 if move_idx < temperature_moves else 0.0
        
        # Add dirichlet noise to root node for exploration
        add_noise = (move_idx < temperature_moves)
        root = mcts.search(board, num_simulations=simulations, add_noise=add_noise)
        
        # Store state for training
        policy_target = np.zeros(4672, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits > 0:
            for move, child in root.children.items():
                idx = move_to_index(move, board)
                prob = child.visit_count / total_visits
                policy_target[idx] = prob
                
        board_array = encode_board_v2(board)
        # We append a placeholder value which will be updated when the game ends
        examples.append((board_array, policy_target, board.turn))
        
        # Check resignation
        if root.q_value < resign_threshold:
            resignation_counter += 1
            if resignation_counter >= resign_count:
                break
        else:
            resignation_counter = 0
            
        move = mcts.select_move(root, temperature=temp)
        if move is None:
            break
        board.push(move)
        
    # Determine the game result
    winner = None
    res = board.result(claim_draw=True)
    if res == '1-0':
        winner = chess.WHITE
    elif res == '0-1':
        winner = chess.BLACK
    elif res == '1/2-1/2':
        winner = None
    else:
        # Resignation case
        if resignation_counter >= resign_count:
            winner = not board.turn
            
    # Assign values to examples (+1 for win, -1 for loss, 0 for draw)
    final_examples = []
    for board_arr, policy, turn in examples:
        if winner is None:
            val = 0.0
        elif winner == turn:
            val = 1.0
        else:
            val = -1.0
        final_examples.append((board_arr, policy, val))
        
    return final_examples

def run_self_play(
    checkpoint_path: str,
    num_games: int,
    simulations: int,
    temperature_moves: int,
    max_moves: int,
    resign_threshold: float,
    resign_count: int,
    device: torch.device,
    c_puct: float,
    dirichlet_alpha: float,
    verbose: bool,
    num_workers: int
):
    examples = []
    device_str = str(device)
    
    start_time = time.time()
    if num_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    play_one_game,
                    checkpoint_path, simulations, temperature_moves, max_moves,
                    resign_threshold, resign_count, device_str, c_puct, dirichlet_alpha
                ) for _ in range(num_games)
            ]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    game_examples = future.result()
                    examples.extend(game_examples)
                    completed += 1
                    if verbose and completed % max(1, num_games // 10) == 0:
                        print(f"  [self-play] {completed}/{num_games} games finished")
                except Exception as e:
                    print(f"Error in self-play worker: {e}")
    else:
        for i in range(num_games):
            try:
                game_examples = play_one_game(
                    checkpoint_path, simulations, temperature_moves, max_moves,
                    resign_threshold, resign_count, device_str, c_puct, dirichlet_alpha
                )
                examples.extend(game_examples)
                if verbose and (i+1) % max(1, num_games // 10) == 0:
                    print(f"  [self-play] {i+1}/{num_games} games finished")
            except Exception as e:
                print(f"Error in self-play loop: {e}")
                
    elapsed = time.time() - start_time
    if verbose:
        print(f"Self-play generated {len(examples)} positions in {elapsed:.1f}s")
        
    return examples

def save_examples(examples: list, out_path: str):
    if not examples:
        return
    boards = np.stack([ex[0] for ex in examples])
    policies = np.stack([ex[1] for ex in examples])
    values = np.array([ex[2] for ex in examples], dtype=np.float32)
    np.savez_compressed(out_path, boards=boards, policies=policies, values=values)

"""
train_joint.py
--------
Joint supervised training of ChessValueNet (Policy + Value Heads) against
Stockfish / puzzle / opening data.

Loss = Policy CrossEntropy (with label smoothing) + Value MSE.

Features:
  - Label smoothing (0.1) to prevent overconfident predictions
  - Gradient clipping to stabilize training on diverse data
  - Resume from existing checkpoint (--resume flag)
  - Warmup + cosine annealing LR schedule

Run:
    python train_joint.py --data data/train_policy_large.npz data/train_puzzles.npz data/train_openings.npz --epochs 30
    python train_joint.py --data data/train_puzzles.npz --epochs 20 --resume models/checkpoints/alphazero_distilled.pt
"""
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import chess
from engine.board_encoder import encode_board_v2
from engine.move_encoding import move_to_index
from models.network import ChessValueNet, count_parameters

CP_SCALE = 100.0  # Scale centipawns for tanh

class JointDataset(Dataset):
    def __init__(self, npz_paths):
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]

        fens_list, evals_list, moves_list = [], [], []
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            fens_list.append(data["fens"])
            evals_list.append(data["evals"].astype(np.float32))
            moves_list.append(data["best_moves"])
            print(f"  loaded {len(data['fens']):,} positions from {path}")

        self.fens = np.concatenate(fens_list)
        self.evals = np.concatenate(evals_list)
        self.moves = np.concatenate(moves_list)

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        x = encode_board_v2(board)
        
        # Value target
        cp = self.evals[idx]
        v_target = np.tanh(cp / CP_SCALE).astype(np.float32)
        
        # Policy target
        best_move = chess.Move.from_uci(self.moves[idx])
        p_target = move_to_index(best_move, board)
        
        return x, p_target, v_target


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_ds = JointDataset(args.data)
    print(f"Loaded {len(full_ds):,} total positions")

    val_size = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize or resume model
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net = ChessValueNet(
            channels=ckpt.get("channels", 128),
            num_res_blocks=ckpt.get("num_res_blocks", 15),
            output_policy=True
        )
        net.load_state_dict(ckpt["model_state_dict"], strict=False)
        net.to(device)
        print(f"  Loaded weights from {args.resume}")
    else:
        print(f"Initializing new ChessValueNet (AlphaZero mode)...")
        net = ChessValueNet(output_policy=True)
        net.to(device)
    
    print(f"Trainable parameters (full network): {count_parameters(net):,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup + Cosine Annealing
    warmup_epochs = min(2, args.epochs // 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Label smoothing prevents overconfident predictions
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss_sum = 0.0
        train_p_loss_sum = 0.0
        train_v_loss_sum = 0.0
        train_acc_sum = 0.0
        n_train = 0
        
        for x, p_targ, v_targ in train_loader:
            x = x.to(device).float()
            p_targ = p_targ.to(device).long()
            v_targ = v_targ.to(device).float()

            optimizer.zero_grad()
            p_logits, v_pred = net(x)
            
            p_loss = ce_loss_fn(p_logits, p_targ)
            v_loss = mse_loss_fn(v_pred, v_targ)
            
            loss = p_loss + v_loss
            loss.backward()
            
            # Gradient clipping to stabilize training on diverse data
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_p_loss_sum += p_loss.item() * x.size(0)
            train_v_loss_sum += v_loss.item() * x.size(0)
            preds = torch.argmax(p_logits, dim=1)
            train_acc_sum += (preds == p_targ).sum().item()
            n_train += x.size(0)

        scheduler.step()

        net.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x, p_targ, v_targ in val_loader:
                x = x.to(device).float()
                p_targ = p_targ.to(device).long()
                v_targ = v_targ.to(device).float()
                
                p_logits, v_pred = net(x)
                
                p_loss = ce_loss_fn(p_logits, p_targ)
                v_loss = mse_loss_fn(v_pred, v_targ)
                
                loss = p_loss + v_loss
                val_loss_sum += loss.item() * x.size(0)
                
                preds = torch.argmax(p_logits, dim=1)
                val_acc_sum += (preds == p_targ).sum().item()
                n_val += x.size(0)

        train_loss = train_loss_sum / max(1, n_train)
        train_p_loss = train_p_loss_sum / max(1, n_train)
        train_v_loss = train_v_loss_sum / max(1, n_train)
        train_acc = train_acc_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        val_acc = val_acc_sum / max(1, n_val)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f} (p={train_p_loss:.3f} v={train_v_loss:.3f})  "
              f"acc={train_acc:.2%}  val_loss={val_loss:.4f} val_acc={val_acc:.2%}  "
              f"lr={current_lr:.6f}  elapsed={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
            torch.save({
                "model_state_dict": net.state_dict(),
                "channels": net.channels,
                "num_res_blocks": net.num_res_blocks,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }, args.out_model)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nTraining done. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2%}")
    print(f"Model saved to {args.out_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", default=["data/train_policy.npz"],
                         help="one or more .npz files containing best_moves and evals")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=256, dest="batch_size")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.1, dest="label_smoothing",
                         help="Label smoothing factor (0=none, 0.1=recommended)")
    parser.add_argument("--patience", type=int, default=8,
                         help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume training from")
    parser.add_argument("--out_model", "--out-model", type=str, default="models/checkpoints/alphazero_distilled.pt", dest="out_model")
    args = parser.parse_args()
    train(args)

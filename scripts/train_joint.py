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
import sys
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import chess
from engine.board_encoder import encode_board_v2
from engine.move_encoding import move_to_index
from models.network import ChessValueNet, count_parameters

CP_SCALE = 400.0  # Scale centipawns for tanh

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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmark autotuner and Automatic Mixed Precision (AMP FP16) for RTX acceleration.")

    full_ds = JointDataset(args.data)
    print(f"Loaded {len(full_ds):,} total positions")

    val_size = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    pin_mem = False  # Disabled on Windows to prevent CUDAPluggableAllocator/CUDAEvent driver crash
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_mem)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0

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
        
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"  Loaded weights from {args.resume} (Completed Epoch {start_epoch - 1}, Best val_loss={best_val_loss:.4f}, Best val_acc={best_val_acc:.2%})")
    else:
        print(f"Initializing new ChessValueNet (AlphaZero mode)...")
        net = ChessValueNet(output_policy=True)
        net.to(device)
    
    print(f"Trainable parameters (full network): {count_parameters(net):,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # Warmup + Cosine Annealing
    warmup_epochs = min(2, args.epochs // 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
    
    # Label smoothing prevents overconfident predictions
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mse_loss_fn = nn.MSELoss()

    patience_counter = 0
    total_batches = len(train_loader)

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs} ({total_batches:,} batches/epoch)...")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        net.train()
        train_loss_sum = 0.0
        train_p_loss_sum = 0.0
        train_v_loss_sum = 0.0
        train_acc_sum = 0.0
        n_train = 0
        
        for batch_idx, (x, p_targ, v_targ) in enumerate(train_loader, 1):
            x = x.to(device).float()
            p_targ = p_targ.to(device).long()
            v_targ = v_targ.to(device).float()

            optimizer.zero_grad()

            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p_logits, v_pred = net(x)
                    p_loss = ce_loss_fn(p_logits, p_targ)
                    v_loss = mse_loss_fn(v_pred, v_targ)
                    loss = p_loss + v_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                p_logits, v_pred = net(x)
                p_loss = ce_loss_fn(p_logits, p_targ)
                v_loss = mse_loss_fn(v_pred, v_targ)
                loss = p_loss + v_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_p_loss_sum += p_loss.item() * x.size(0)
            train_v_loss_sum += v_loss.item() * x.size(0)
            preds = torch.argmax(p_logits, dim=1)
            train_acc_sum += (preds == p_targ).sum().item()
            n_train += x.size(0)

            # Live batch progress update every 200 batches
            if batch_idx % 200 == 0 or batch_idx == total_batches:
                curr_loss = train_loss_sum / n_train
                curr_acc = train_acc_sum / n_train
                print(f"\r  Epoch {epoch:2d}/{args.epochs} | Batch {batch_idx:5d}/{total_batches:5d} ({batch_idx/total_batches:5.1%}) | Loss: {curr_loss:.4f} | Acc: {curr_acc:.2%}", end="", flush=True)

        scheduler.step()
        print()  # Newline after training batches

        net.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x, p_targ, v_targ in val_loader:
                x = x.to(device).float()
                p_targ = p_targ.to(device).long()
                v_targ = v_targ.to(device).float()
                
                if scaler:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        p_logits, v_pred = net(x)
                        p_loss = ce_loss_fn(p_logits, p_targ)
                        v_loss = mse_loss_fn(v_pred, v_targ)
                        loss = p_loss + v_loss
                else:
                    p_logits, v_pred = net(x)
                    p_loss = ce_loss_fn(p_logits, p_targ)
                    v_loss = mse_loss_fn(v_pred, v_targ)
                    loss = p_loss + v_loss
                val_loss_sum += loss.item() * x.size(0)
                
                preds = torch.argmax(p_logits, dim=1)
                val_acc_sum += (preds == p_targ).sum().item()
                n_val += x.size(0)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss = train_loss_sum / max(1, n_train)
        train_p_loss = train_p_loss_sum / max(1, n_train)
        train_v_loss = train_v_loss_sum / max(1, n_train)
        train_acc = train_acc_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        val_acc = val_acc_sum / max(1, n_val)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.time() - epoch_t0
        
        print(f"Summary {epoch:2d}/{args.epochs} -> loss={train_loss:.4f} (p={train_p_loss:.3f} v={train_v_loss:.3f}) | "
              f"acc={train_acc:.2%} | val_loss={val_loss:.4f} | val_acc={val_acc:.2%} | "
              f"lr={current_lr:.6f} | time={epoch_elapsed:.1f}s")

        if val_loss < best_val_loss or val_acc > best_val_acc:
            improved = []
            if val_loss < best_val_loss:
                improved.append(f"val_loss: {best_val_loss:.4f} -> {val_loss:.4f}")
                best_val_loss = val_loss
                patience_counter = 0
            if val_acc > best_val_acc:
                improved.append(f"val_acc: {best_val_acc:.2%} -> {val_acc:.2%}")
                best_val_acc = val_acc
                patience_counter = 0

            os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
            save_payload = {
                "model_state_dict": net.state_dict(),
                "channels": net.channels,
                "num_res_blocks": net.num_res_blocks,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            torch.save(save_payload, args.out_model)
            best_path = os.path.join(os.path.dirname(args.out_model), "alphazero_best.pt")
            if os.path.abspath(args.out_model) != os.path.abspath(best_path):
                torch.save(save_payload, best_path)
            print(f"  --> [BEST MODEL SAVED] ({', '.join(improved)}) -> {args.out_model}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (no validation improvement for {args.patience} epochs).")
                print(f"Prevented overfitting! Your best saved model is safe at: {args.out_model}")
                break

        # Save latest checkpoint every epoch for seamless recovery
        latest_path = os.path.join(os.path.dirname(args.out_model), "alphazero_latest.pt")
        torch.save({
            "model_state_dict": net.state_dict(),
            "channels": net.channels,
            "num_res_blocks": net.num_res_blocks,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch,
        }, latest_path)

    print(f"\n{'='*60}")
    print(f"Training Complete! Best val_loss={best_val_loss:.4f}, Best val_acc={best_val_acc:.2%}")
    print(f"Saved Checkpoint: {args.out_model}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", default=["data/train_elite_games.npz", "data/train_puzzles_large.npz"],
                         help="one or more .npz files containing best_moves and evals")
    parser.add_argument("--epochs", type=int, default=40,
                         help="Total training epochs (default: 40 with cosine decay)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=512, dest="batch_size")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0, dest="num_workers",
                         help="DataLoader worker processes (default: 0 for Windows in-memory stability)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.1, dest="label_smoothing",
                         help="Label smoothing factor (0=none, 0.1=recommended)")
    parser.add_argument("--patience", type=int, default=10,
                         help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume training from")
    parser.add_argument("--out_model", "--out-model", type=str, default="models/checkpoints/alphazero_distilled.pt", dest="out_model")
    args = parser.parse_args()
    train(args)

"""
scripts/eval_suite.py
---------------------
Runs head-to-head matches:
1. alphazero_1.pt vs alphazero_distilled.pt (4 games)
2. alphazero_2.pt vs alphazero_distilled.pt (4 games)
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PYTHON_EXE = sys.executable

def run_match(challenger_path: str, pgn_path: str, challenger_name: str):
    print("\n" + "=" * 70)
    print(f"   STARTING MATCH: {challenger_name} vs alphazero_distilled.pt (4 games)")
    print("=" * 70)
    cmd = [
        PYTHON_EXE,
        os.path.join(SCRIPT_DIR, "match_eval.py"),
        "--base-checkpoint", os.path.join(ROOT_DIR, "models", "checkpoints", "alphazero_distilled.pt"),
        "--rl-checkpoint", challenger_path,
        "--games", "4",
        "--simulations", "100",
        "--opening-plies", "4",
        "--pgn-out", pgn_path
    ]
    proc = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=False)
    return proc.returncode

def main():
    ck_1 = os.path.join(ROOT_DIR, "models", "checkpoints", "alphazero_1.pt")
    ck_2 = os.path.join(ROOT_DIR, "models", "checkpoints", "alphazero_2.pt")
    pgn_1 = os.path.join(ROOT_DIR, "models", "checkpoints", "h2h_distilled_vs_1.pgn")
    pgn_2 = os.path.join(ROOT_DIR, "models", "checkpoints", "h2h_distilled_vs_2.pgn")

    print(f"Evaluating {ck_1} and {ck_2} against our current best (alphazero_distilled.pt)...")
    run_match(ck_1, pgn_1, "alphazero_1.pt (15-ResBlock, Aug 29)")
    run_match(ck_2, pgn_2, "alphazero_2.pt (10-ResBlock, RL Iter 2)")
    print("\n" + "=" * 70)
    print("   ALL TOURNAMENT MATCHES COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# Allow running scripts without installing the package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdpnet.train import train


def main():
    p = argparse.ArgumentParser(description="Train RDPNet on Wafer dataset")
    p.add_argument("--data", default="Wafer_Map_Datasets.npz", help="Path to dataset .npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--asl", action="store_true", help="Use Asymmetric Loss instead of BCE")
    p.add_argument("--no-dpm", action="store_true", help="Baseline CNN: remove every DPM")
    p.add_argument("--fp32", action="store_true", help="Train in full precision (slower, no fp16 overflow)")
    p.add_argument("--ckpt-dir", default="checkpoint")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=3529, help="Training seed (the data split stays fixed)")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = train(
        npz_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_asl=args.asl,
        use_dpm=not args.no_dpm,
        use_amp=not args.fp32,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        seed=args.seed,
        device=device,
    )
    print(f"Best val acc: {res['best_acc']:.6f} @ epoch {res['best_epoch']}")


if __name__ == "__main__":
    main()

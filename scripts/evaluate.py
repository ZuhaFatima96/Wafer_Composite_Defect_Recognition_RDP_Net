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

from rdpnet.data import make_dataloaders
from rdpnet.eval import evaluate
from rdpnet.models.rdpnet import RDPNet


def main():
    p = argparse.ArgumentParser(description="Evaluate RDPNet on the test split")
    p.add_argument("--data", default="Wafer_Map_Datasets.npz", help="Path to dataset .npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint .mdl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-dpm", action="store_true", help="Use this for checkpoints trained with --no-dpm")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = make_dataloaders(args.data, batch_size=args.batch_size)

    model = RDPNet(output_dim=8, in_channels=1, apply_sigmoid=False, use_dpm=not args.no_dpm).to(device)  # outputs logits
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    acc, details = evaluate(
        model,
        test_loader,
        device=device,
        threshold=args.threshold,
        return_details=True,
        from_logits=True,
        use_amp=False,  # the final test numbers are always computed in full precision
    )
    print(f"Test acc (mean over C1..C38): {acc:.6f}")
    print(f"Precision mean: {details['precision_mean']:.3f}")
    print(f"Recall mean:    {details['recall_mean']:.3f}")
    print(f"F1 mean:        {details['f1_mean']:.3f}")


if __name__ == "__main__":
    main()

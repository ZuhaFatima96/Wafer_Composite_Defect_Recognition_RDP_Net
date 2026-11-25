import csv
import os
from typing import Optional, Dict, Any

import torch
from torch import nn, optim
from tqdm import tqdm

from .data import make_dataloaders
from .eval import evaluate
from .losses import AsymmetricLoss
from .models.rdpnet import RDPNet
from .utils import set_seed


def train(
    npz_path: str = "Wafer_Map_Datasets.npz",
    output_dim: int = 8,
    in_channels: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 100,
    device: Optional[torch.device] = None,
    use_asl: bool = False,
    ckpt_dir: str = "checkpoint",
    resume: Optional[str] = None,
    seed: int = 3529,
) -> Dict[str, Any]:
    set_seed(seed)
    torch.backends.cudnn.benchmark = True  # let cuDNN pick the fastest convolution algorithms
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_dataloaders(npz_path, batch_size=batch_size)

    # the model always outputs logits; the sigmoid happens inside the loss and in evaluate()
    model = RDPNet(output_dim=output_dim, in_channels=in_channels, apply_sigmoid=False).to(device)
    criterion = AsymmetricLoss() if use_asl else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler()  # mixed precision: scales the loss so small fp16 gradients don't round to zero

    if resume and os.path.isfile(resume):
        model.load_state_dict(torch.load(resume, map_location=device))

    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_acc", "test_acc"])

    best_acc, best_epoch = -1.0, -1  # start below 0 so best.mdl is always saved at least once
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            x = x.unsqueeze(1).to(device)
            y = y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):  # forward pass in half precision
                logits = model(x)
            loss = criterion(logits.float(), y)  # loss in full precision

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # the model is chosen on validation; test accuracy is only logged, never used for choosing
        val_acc, _ = evaluate(model, val_loader, device=device, from_logits=True)
        test_acc, _ = evaluate(model, test_loader, device=device, from_logits=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss.item(), val_acc, test_acc])

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.mdl"))
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.mdl"))

    return {"best_acc": best_acc, "best_epoch": best_epoch, "ckpt_dir": ckpt_dir}

#!/usr/bin/env python3
"""train.py
===========
Train hypar‑pressure operator networks (FNN / CNN / DeepONet).

Usage
-----
```bash
python -m src.train --model deeponet --epochs 4000 --batch 8 --device cuda:0
```
"""
from __future__ import annotations

import argparse, csv, math, time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---------------- project imports -----------------------------------------
from src.dataloader import get_dataloaders
from src.models.fnn import FNN
from src.models.cnn import CNN
from src.models.deeponet import DeepONet


MODEL_MAP = {
    "fnn": FNN,
    "cnn": CNN,
    "deeponet": DeepONet,
}

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hypar operator NN")
    p.add_argument("--model", choices=MODEL_MAP.keys(), required=True)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=400,
                   help="early‑stop patience (epochs)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _save_loss(hist: list[float], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss"])
        wr.writerows(enumerate(hist, 1))

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    args = _parse()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    root = Path(__file__).resolve().parents[1]

    # ---------- data ----------
    train_dl, test_dl, scaler = get_dataloaders(
        npz_path=root / "data" / "dataset.npz",
        batch_size=args.batch,
        seed=args.seed,
    )

    # ---------- model ----------
    Net = MODEL_MAP[args.model]
    model = Net().to(device)

    mse = nn.MSELoss()
    l1  = nn.L1Loss()

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = root / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    stall = 0
    history: list[float] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        with tqdm(train_dl, desc=f"{args.model.upper()} {ep}/{args.epochs}") as it:
            for batch in it:
                x = batch["x"].to(device)
                p0_gt = batch["p0"].to(device)
                ppk_gt = batch["ppk"].to(device)
                dt_gt = batch["dt"].to(device)

                opt.zero_grad()

                if args.model == "fnn":
                    p0_pred, ppk_pred, dt_pred = model(x)
                elif args.model == "deeponet":
                    # `grid` is identical for every sample; take first row to get (900,2)
                    grid = batch["grid"][0].to(device)
                    maps_pred, dt_pred = model(x, grid)
                    p0_pred, ppk_pred = maps_pred[:, 0:1], maps_pred[:, 1:2]
                elif args.model == "cnn":
                    maps_pred, dt_pred = model(x)
                    p0_pred, ppk_pred = maps_pred[:, 0:1], maps_pred[:, 1:2]

                loss = mse(p0_pred, p0_gt) + mse(ppk_pred, ppk_gt) + l1(dt_pred, dt_gt)
                loss.backward()
                opt.step()

                running += loss.item() * x.size(0)
                it.set_postfix(loss=loss.item())

        train_loss = running / len(train_dl.dataset)
        history.append(train_loss)
        sched.step()

        # early‑stopping on training loss
        if train_loss < best_loss - 1e-6:
            best_loss = train_loss
            stall = 0
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
            }, out_dir / "model.pt")
        else:
            stall += 1
            if stall >= args.patience:
                print(f"Early stop at epoch {ep}")
                break

    _save_loss(history, out_dir / "loss.csv")
    print(f"Best training loss = {best_loss:.4e}")


if __name__ == "__main__":
    main()

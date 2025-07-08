#!/usr/bin/env python3
"""
evaluate.py

Compute test-set metrics and generate publishable figures comparing
FNN, CNN, and DeepONet on the hypar-pressure dataset.
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.dataloader import get_dataloaders
from src.models.fnn import FNN
from src.models.cnn import CNN
from src.models.deeponet import DeepONet

# Map names → classes & result directories
MODELS = {
    "fnn": (FNN,   Path(__file__).resolve().parents[1] / "results" / "fnn"),
    "cnn": (CNN,   Path(__file__).resolve().parents[1] / "results" / "cnn"),
    "deeponet": (DeepONet,
                 Path(__file__).resolve().parents[1] / "results" / "deeponet"),
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models")
    p.add_argument("--batch", type=int, default=1,
                   help="batch size for inference")
    p.add_argument("--device", default="cpu",
                   help="torch device for inference")
    return p.parse_args()

def load_checkpoint(model_cls, ckpt_path, device):
    """Instantiate model, load state_dict + scaler."""
    ck = torch.load(ckpt_path, map_location=device)
    model = model_cls().to(device).eval()
    model.load_state_dict(ck["model_state"])
    scaler = ck.get("scaler", ck.get("input_mean", None))  # handle scalar naming
    # If scaler was saved as dict, extract the mean/std
    if isinstance(scaler, dict):
        scalar_mean = scaler["mean"]
        scalar_std  = scaler["std"]
    else:
        scalar_mean = ck.get("input_mean", None)
        scalar_std  = ck.get("input_std",  None)
    return model, scalar_mean, scalar_std

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    r2  = r2_score(y_true, y_pred)
    return mse, r2

def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Build test loader (use same split logic) ---
    root = Path(__file__).resolve().parents[1]
    train_dl, test_dl, _ = get_dataloaders(
        npz_path=root / "data" / "dataset.npz",
        batch_size=args.batch,
        test_frac=0.20,
        seed=42,
    )

    # Containers for metrics & plots
    metrics = {}
    dt_preds = {}
    dt_trues = []
    loss_histories = {}

    # --- Loop over models ---
    for name, (cls, res_dir) in MODELS.items():
        # 1) Load checkpoint + loss history
        ckpt = res_dir / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found. Train model '{name}' first.")
        model, _, _ = load_checkpoint(cls, ckpt, device)

        # 2) Read loss.csv
        lh = np.loadtxt(res_dir / "loss.csv", delimiter=",", skiprows=1)
        epochs, losses = lh[:,0], lh[:,1]
        loss_histories[name] = (epochs, losses)

        # 3) Run inference on test set
        all_p0_gt, all_ppk_gt, all_dt_gt = [], [], []
        all_p0_pr, all_ppk_pr, all_dt_pr = [], [], []
        for batch in test_dl:
            x = batch["x"].to(device)
            p0 = batch["p0"].numpy().reshape(-1)   # flatten
            ppk= batch["ppk"].numpy().reshape(-1)
            dt = batch["dt"].numpy().reshape(-1)
            # forward
            if name == "deeponet":
                grid = batch["grid"][0].to(device)
                maps, dt_pred = model(x, grid)
            else:
                maps, dt_pred = model(x)
            p0_pred  = maps[:,0:1].detach().cpu().numpy().reshape(-1)
            ppk_pred = maps[:,1:2].detach().cpu().numpy().reshape(-1)
            dt_pred  = dt_pred.detach().cpu().numpy().reshape(-1)

            all_p0_gt .append(p0)
            all_ppk_gt.append(ppk)
            all_dt_gt .append(dt)
            all_p0_pr .append(p0_pred)
            all_ppk_pr.append(ppk_pred)
            all_dt_pr .append(dt_pred)

        # concatenate
        p0_gt  = np.concatenate(all_p0_gt)
        ppk_gt = np.concatenate(all_ppk_gt)
        dt_gt  = np.concatenate(all_dt_gt)
        p0_pr  = np.concatenate(all_p0_pr)
        ppk_pr = np.concatenate(all_ppk_pr)
        dt_pr  = np.concatenate(all_dt_pr)

        # store for parity plot
        dt_preds[name] = (dt_gt, dt_pr)
        if not dt_trues:
            dt_trues = dt_gt

        # compute metrics
        m_p0,  r2_p0  = compute_metrics(p0_gt,  p0_pr)
        m_ppk, r2_ppk = compute_metrics(ppk_gt, ppk_pr)
        m_dt,  r2_dt  = compute_metrics(dt_gt,  dt_pr)
        metrics[name] = (m_p0, r2_p0, m_ppk, r2_ppk, m_dt, r2_dt)

    # --- 1) Print LaTeX table ---
    print("\n% LaTeX table of test metrics")
    print("\\begin{tabular}{l|rr|rr|rr}")
    print("Model & MSE$_{P0}$ & $R^2_{P0}$ & MSE$_{Ppk}$ & $R^2_{Ppk}$ & MSE$_{\\Delta t}$ & $R^2_{\\Delta t}$ \\\\")
    print("\\hline")
    for name, vals in metrics.items():
        print(f"{name}  & " +
              " & ".join(f"{v:.2e}" for v in vals) + " \\\\")
    print("\\end{tabular}\n")

    # --- 2) Plot loss curves ---
    plt.figure(figsize=(6,4))
    for name, (ep, ls) in loss_histories.items():
        plt.plot(ep, ls, label=name.upper(), marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curves")
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "results" / "loss_curves.png", dpi=300)
    plt.show()

    # --- 3) Parity scatter for Delta t ---
    plt.figure(figsize=(5,5))
    for name, (true, pred) in dt_preds.items():
        plt.scatter(true, pred, s=20, alpha=0.7, label=name.upper())
    lims = [min(dt_trues), max(dt_trues)]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel("True Δt [s]")
    plt.ylabel("Predicted Δt [s]")
    plt.title("Parity Plot of Δt")
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "results" / "parity_dt.png", dpi=300)
    plt.show()

    # --- 4) Pressure-map comparisons for 3 samples ---
    # pick first 3 test indices
    test_indices = test_dl.dataset.indices[:3]
    for idx in test_indices:
        # reload raw data from npz
        data = np.load(root / "data" / "dataset.npz")
        Rn, dr, H = data["Rn"][idx], data["dr"][idx], data["H"][idx]
        gt_p0   = data["P_t0"][idx]
        gt_ppk  = data["P_tpeak"][idx]

        fig, axs = plt.subplots(2, len(MODELS), figsize=(4*len(MODELS), 8))
        for col, (name, (cls, res_dir)) in enumerate(MODELS.items()):
            # load model
            model, *_ = load_checkpoint(cls, res_dir / "model.pt", device)
            # prepare input x and (grid)
            scalars = torch.tensor([(Rn, dr, H)],
                                   dtype=torch.float32).to(device)
            if name == "deeponet":
                grid = test_dl.dataset._get_unit_grid().to(device)
                maps, _ = model(scalars, grid)
            else:
                maps, _ = model(scalars)
            pr_p0  = maps[0:1,0:1].detach().cpu().numpy().reshape(30,30)
            pr_ppk = maps[0:1,1:2].detach().cpu().numpy().reshape(30,30)

            im0 = axs[0,col].imshow(gt_p0,  origin='lower')
            axs[0,col].set_title(f"{name.upper()} GT $P_0$")
            plt.colorbar(im0, ax=axs[0,col], fraction=0.046)

            im1 = axs[1,col].imshow(pr_p0, origin='lower')
            axs[1,col].set_title(f"{name.upper()} Pred $P_0$")
            plt.colorbar(im1, ax=axs[1,col], fraction=0.046)

        plt.suptitle(f"Sample {idx}  |  Rn={Rn:.2f}, dr={dr:.2f}, H={H:.2f}")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(root / f"results/compare_sample_{idx}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    main()

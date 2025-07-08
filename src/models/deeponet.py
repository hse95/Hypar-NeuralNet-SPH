"""models/deeponet.py
====================
Branch–Trunk **DeepONet** adapted to the hypar‑pressure dataset.

Inputs
------
* **Branch**   : 3 normalised scalars `[Rn, dr, H]`  → latent (256)
* **Trunk**    : (900,2) grid coordinates in [−1,1]²  → latent (128)

Outputs
-------
* Two (30 × 30) pressure maps  (P₀, Pₚₖ)
* One scalar Δt

Dot‑product decoding: branch latent is split into two 128‑dim halves used as
weights for P₀ and Pₚₖ respectively; trunk latent acts as a set of basis
functions over space. Δt is predicted by a small MLP from the full branch
latent.

This design keeps parameter count low (≈100k) and leverages operator learning
strengths of DeepONet while suiting the small (125) dataset.
"""
from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_B = 256  # branch output
LATENT_T = 128  # trunk output


def _mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Branch‑Trunk DeepONet for 30×30 pressure maps + Δt."""

    def __init__(self, hidden_branch: int = 128, hidden_trunk: int = 128,
                 depth: int = 4):
        super().__init__()
        # Branch: scalars → latent 256 (split into two 128‑chunks)
        self.branch = _mlp(3, LATENT_B, hidden_branch, depth)

        # Trunk: (x,y) → latent 128 shared for all samples (runs once / fwd)
        self.trunk = _mlp(2, LATENT_T, hidden_trunk, depth)

        # Δt head
        self.dt_head = nn.Sequential(
            nn.Linear(LATENT_B, 128), nn.GELU(), nn.Linear(128, 1)
        )

    # ------------------------------------------------------------------
    def forward(self, x_scalar: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        x_scalar : (B,3) float32, normalised scalars.
        grid     : (900,2) float32, constant spatial coordinates in [‑1,1]².

        Returns
        -------
        maps : (B,2,30,30) tensor – P₀ and Pₚₖ.
        dt   : (B,1) tensor – Δt.
        """
        B = x_scalar.size(0)

        # Branch latent (B,256)
        b = self.branch(x_scalar)
        b1, b2 = b[:, :LATENT_T], b[:, LATENT_T:]   # each (B,128)

        # Trunk latent (900,128) – computed once; share across batch
        t = self.trunk(grid)                        # (P,128)

        # Dot products → (B,P)
        p0_flat  = torch.einsum('bl,pl->bp', b1, t)
        ppk_flat = torch.einsum('bl,pl->bp', b2, t)

        # Reshape to maps (B,1,30,30)
        P0  = p0_flat.view(B, 1, 30, 30)
        Ppk = ppk_flat.view(B, 1, 30, 30)

        # Δt scalar
        dt = self.dt_head(b)
        maps = torch.cat([P0, Ppk], dim=1)  # (B,2,30,30)
        return maps, dt

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(pred: Tuple[torch.Tensor, torch.Tensor],
                     target_maps: torch.Tensor,
                     target_dt: torch.Tensor,
                     w_maps: float = 1.0,
                     w_dt: float = 1.0) -> Dict[str, torch.Tensor]:
        maps_pred, dt_pred = pred
        loss_maps = F.mse_loss(maps_pred, target_maps)
        loss_dt   = F.l1_loss(dt_pred, target_dt)
        total = w_maps * loss_maps + w_dt * loss_dt
        return {"total": total, "maps": loss_maps, "dt": loss_dt}


# Register
MODEL_NAME = "deeponet"

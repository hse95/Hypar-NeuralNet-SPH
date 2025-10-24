"""models/fnn.py
================
Simple fully‑connected network that maps **3 scalar inputs** → two 30×30
pressure maps + one scalar Δt.

Architecture
------------
[Rn, dr, H] → Linear(3→256) → GELU → (×3 hidden) → Linear → 1801 outputs
(= 30×30×2 + 1) which are reshaped in `forward()`.

* Uses `nn.Flatten()` only at the output side to keep code explicit.
* Weight initialisation: Kaiming‑uniform for ReLU/GELU layers.
"""

from typing import Tuple
import torch
import torch.nn as nn

OUTPUT_P_SIZE = 30 * 30 * 2 + 1  # 2 pressure maps + Δt scalar


def kaiming_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FNN(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, OUTPUT_P_SIZE))
        self.net = nn.Sequential(*layers)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3) tensor of normalised scalars  [Rn, dr, H]

        Returns
        -------
        P0, Ppk : tensors (B, 1, 30, 30)
        dt       : tensor  (B, 1)
        """
        out = self.net(x)                              # (B, 1801)
        p0_flat  = out[:, :900]
        ppk_flat = out[:, 900:1800]
        dt       = out[:, -1:].contiguous()
        # reshape to (B, 1, 30, 30)
        P0  = p0_flat.view(-1, 1, 30, 30)
        Ppk = ppk_flat.view(-1, 1, 30, 30)
        return P0, Ppk, dt

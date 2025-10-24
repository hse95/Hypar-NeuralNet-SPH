"""models/cnn.py
================
A compact **encoder–decoder CNN** for predicting two pressure maps and one
scalar Δt from the three scalar inputs `[Rn, dr, H]`.

Key points
----------
* **No external base‑class** – this is a plain `torch.nn.Module`, so importing
  it in `train.py` will not raise unresolved‑symbol errors.
* Hidden width kept modest (≤64 channels) to avoid over‑fit on 100‑sample
  training set.
* `compute_loss()` mirrors the signature used in FNN, returning a dict with
  components for easy logging.

Usage example
-------------
```python
from models.cnn import CNN
model = CNN()
map_out, dt_out = model(x_scalar)  # x_scalar : (B,3)
```
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """Encoder–decoder CNN mapping (B,3) → (B,2,30,30) maps + (B,1) scalar."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Dense stem: 3 scalars → 5×5×hidden_dim latent volume
        self.dense_stem = nn.Sequential(
            nn.Linear(3, 5 * 5 * hidden_dim),
            nn.GELU()
        )

        # Decoder: ConvTranspose2d blocks to scale 5×5 → 30×30
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim // 2)
        )  # 5×5 → 10×10

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim // 4)
        )  # 10×10 → 20×20

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim // 4)
        )  # 20×20 → 40×40 (we will crop later)

        # Final 1×1 conv to 2 channels then center‑crop to 30×30
        self.map_head = nn.Conv2d(hidden_dim // 4, 2, kernel_size=1)

        # Δt scalar head (shared global latent vector)
        self.dt_head = nn.Sequential(
            nn.Linear(5 * 5 * hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        x : torch.Tensor
            shape (B,3) containing [Rn, dr, H]. Must be float32.

        Returns
        -------
        maps : torch.Tensor
            shape (B,2,30,30) – P_t0 and P_tpeak.
        dt   : torch.Tensor
            shape (B,1) – Δt scalar.
        """
        B = x.size(0)
        latent = self.dense_stem(x)                   # (B, 5*5*hid)
        dt = self.dt_head(latent)                    # (B,1)

        feat = latent.view(B, -1, 5, 5)              # (B, hid, 5,5)
        feat = self.up1(feat)                        # 10×10
        feat = self.up2(feat)                        # 20×20
        feat = self.up3(feat)                        # 40×40

        # Center‑crop 40×40 → 30×30
        start = (feat.size(2) - 30) // 2  # =5
        feat = feat[:, :, start:start+30, start:start+30]
        maps = self.map_head(feat)                   # (B,2,30,30)
        return maps, dt

    # ------------------------------------------------------------------
    # Loss helper
    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(pred: Tuple[torch.Tensor, torch.Tensor],
                     target_maps: torch.Tensor,
                     target_dt: torch.Tensor,
                     w_maps: float = 1.0,
                     w_dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """Return dict of loss components + total."""
        maps_pred, dt_pred = pred
        loss_maps = F.mse_loss(maps_pred, target_maps)
        loss_dt   = F.l1_loss(dt_pred, target_dt)
        total = w_maps * loss_maps + w_dt * loss_dt
        return {"total": total, "maps": loss_maps, "dt": loss_dt}


# Register for dynamic import in train.py
MODEL_NAME = "cnn"

"""dataloader.py
================
Robust data pipeline for hypar‑pressure operator learning (train / test split).

Fixes
-----
* Stratified splitter now guarantees **at least one train sample** for every
  (Rn, dr, H) combo (avoids empty train set when a combo had only one sample).
* Index arrays are cast to `int64` before indexing, preventing the earlier
  `IndexError`.

Key features (unchanged)
------------------------
* 80 % train / 20 % test split (stratified across 5×5×5 grid).
* Z‑score normalisation of `[Rn, dr, H]` using *train* statistics.
* Constant (900,2) grid for operator nets.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Grid helper (cached)
# ---------------------------------------------------------------------------
_COORD_CACHE: torch.Tensor | None = None

def _get_unit_grid() -> torch.Tensor:
    global _COORD_CACHE
    if _COORD_CACHE is None:
        xs = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        _COORD_CACHE = torch.from_numpy(np.stack([xv, yv], -1).reshape(-1, 2))
    return _COORD_CACHE

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HyparPressureDataset(Dataset):
    def __init__(self, npz_path: Path, idx: np.ndarray,
                 mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.data = np.load(npz_path)
        self.idx = idx.astype(np.int64)   # ensure int64 indices
        self.mean = mean; self.std = std
        self._p0 = self.data["P_t0"].astype(np.float32)
        self._ppk = self.data["P_tpeak"].astype(np.float32)
        self._dt = self.data["dT"].astype(np.float32)
        self._Rn, self._dr, self._H = self.data["Rn"], self.data["dr"], self.data["H"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        scalars = np.array([self._Rn[j], self._dr[j], self._H[j]], dtype=np.float32)
        scalars = (scalars - self.mean) / self.std
        return {
            "x"   : torch.from_numpy(scalars),            # (3,)
            "grid": _get_unit_grid(),                    # (900,2)
            "p0"  : torch.from_numpy(self._p0[j][None]), # (1,30,30)
            "ppk" : torch.from_numpy(self._ppk[j][None]),
            "dt"  : torch.tensor([self._dt[j]], dtype=torch.float32),
        }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(npz_path: Path, batch_size: int = 16, test_frac: float = 0.20,
                    seed: int = 42, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    data = np.load(npz_path)
    N = data["Rn"].shape[0]

    # -- label each combo -----------------------------------------------
    def _label(arr):
        uniq = np.unique(arr); return np.searchsorted(uniq, arr)
    combo = _label(data["Rn"]) * 25 + _label(data["dr"]) * 5 + _label(data["H"])

    train_idx, test_idx = [], []
    for c in np.unique(combo):
        idx_all = np.where(combo == c)[0]
        rng.shuffle(idx_all)
        n_total = len(idx_all)
        n_test = max(1, math.floor(test_frac * n_total))
        if n_test == n_total:  # keep at least 1 train sample
            n_test -= 1
        test_idx.extend(idx_all[:n_test])
        train_idx.extend(idx_all[n_test:])

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx  = np.array(test_idx,  dtype=np.int64)

    # -- scaler based on TRAIN -----------------------------------------
    scalars_train = np.stack([data["Rn"][train_idx], data["dr"][train_idx], data["H"][train_idx]], -1)
    mean = scalars_train.mean(0)
    std  = scalars_train.std(0) + 1e-8
    scaler = {"mean": mean, "std": std}

    train_ds = HyparPressureDataset(npz_path, train_idx, mean, std)
    test_ds  = HyparPressureDataset(npz_path, test_idx,  mean, std)

    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers, generator=g)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
    return train_dl, test_dl, scaler

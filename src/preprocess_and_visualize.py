#!/usr/bin/env python3
"""
preprocess_and_visualize.py
===========================
End-to-end tool that **(1) builds the dataset** from all CSV files in
*../data/* and **(2) launches an interactive plot** so you can inspect the
automatically picked t₀ / t_peak for every sample.
"""

from __future__ import annotations
import os, re, sys
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    sys.exit("SciPy is required →  pip install scipy")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# DATA_FOLDER = "../data"          # CSV directory
DATA_FOLDER = r"D:\OneDrive - Princeton University\ResearchWork\Papers\03 - NN_for_FSI\analysis\data"
OUTPUT_FILE = "dataset.npz"      # archive name

# Detection parameters
CUTOFF_HZ      = 0.4
FILTER_ORDER   = 5
BASELINE_T_MAX = 5.0               # seconds
THRESH_MULT    = 1.05              # 5 %
OFFSET_Pa      = 0.5e7             # fictitious boost if baseline too small
EXPECTED_COLS  = 900               # pressure columns per row

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def butter_lowpass(y: np.ndarray, fs: float,
                   fc: float = CUTOFF_HZ,
                   order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, fc / nyq, btype="low", analog=False)
    return filtfilt(b, a, y, padlen=3 * (max(len(b), len(a)) - 1))


def parse_filename(fname: str) -> Tuple[float, float, float]:
    pat = r"Rn(\d+(?:\.\d+)?)dr(\d+(?:\.\d+)?)H(\d+(?:\.\d+)?)m"
    m = re.search(pat, fname)
    if m is None:
        raise ValueError(f"Filename '{fname}' does not match pattern '{pat}'")
    rn_s, dr_s, h_s = m.groups()
    rn = float(rn_s) / 1000 if "." not in rn_s else float(rn_s)
    dr = float(dr_s) / 100  if "." not in dr_s else float(dr_s)
    return rn, dr, float(h_s)


def compute_t0_tpeak(time: np.ndarray,
                     tot_raw: np.ndarray) -> Tuple[int, int]:
    fs = 1.0 / np.median(np.diff(time))
    tot_filt = butter_lowpass(tot_raw, fs)

    baseline_mask = (time >= 0.0) & (time < BASELINE_T_MAX)
    baseline_mean = tot_filt[baseline_mask].mean()
    if baseline_mean < OFFSET_Pa:
        tot_filt = tot_filt + OFFSET_Pa
        baseline_mean += OFFSET_Pa

    thresh = baseline_mean * THRESH_MULT
    mask = (time >= BASELINE_T_MAX) & (tot_filt > thresh)
    idx_candidates = np.where(mask)[0]
    if idx_candidates.size == 0:
        raise ValueError("t₀ not found – signal may be abnormal")
    idx_t0  = int(idx_candidates[0])
    idx_peak = int(np.argmax(tot_raw))
    return idx_t0, idx_peak

# ---------------------------------------------------------------------------
# 1) PRE-PROCESS ALL FILES
# ---------------------------------------------------------------------------
print("[INFO] Preprocessing CSV files …")
all_Rn, all_dr, all_H = [], [], []
P_t0_list, P_tpeak_list, dT_list = [], [], []

csv_files = [f for f in sorted(os.listdir(DATA_FOLDER))
             if f.lower().endswith(".csv")]
if not csv_files:
    sys.exit("No CSV files found in ../data/")

TIME_ALL, TOT_RAW_ALL, TOT_FILT_ALL, IDX_T0_ALL, IDX_PK_ALL = [], [], [], [], []

for fname in csv_files:
    Rn_val, dr_val, H_val = parse_filename(fname)
    csv_path = os.path.join(DATA_FOLDER, fname)
    df = pd.read_csv(csv_path, delimiter=";", skiprows=3)

    time  = df.iloc[:, 1].astype(float).values
    press = df.iloc[:, 2:].to_numpy()
    if press.shape[1] != EXPECTED_COLS:
        raise ValueError(f"{fname}: expected {EXPECTED_COLS} pressure columns, "
                         f"got {press.shape[1]}")

    tot_raw = press.sum(axis=1)
    idx_t0, idx_pk = compute_t0_tpeak(time, tot_raw)

    P_t0_list.append(press[idx_t0].reshape(30, 30).astype(np.float32))
    P_tpeak_list.append(press[idx_pk].reshape(30, 30).astype(np.float32))
    dT_list.append(np.float32(time[idx_pk] - time[idx_t0]))

    all_Rn.append(Rn_val)
    all_dr.append(dr_val)
    all_H.append(H_val)

    TIME_ALL.append(time)
    TOT_RAW_ALL.append(tot_raw)
    fs = 1.0 / np.median(np.diff(time))
    TOT_FILT_ALL.append(butter_lowpass(tot_raw, fs))
    IDX_T0_ALL.append(idx_t0)
    IDX_PK_ALL.append(idx_pk)

np.savez(
    OUTPUT_FILE,
    Rn=np.array(all_Rn, dtype=np.float32),
    dr=np.array(all_dr, dtype=np.float32),
    H=np.array(all_H, dtype=np.float32),
    P_t0=np.stack(P_t0_list),
    P_tpeak=np.stack(P_tpeak_list),
    dT=np.array(dT_list, dtype=np.float32),
)
print(f"[INFO] Saved dataset → {OUTPUT_FILE}  (N={len(all_Rn)})")

# ---------------------------------------------------------------------------
# 2) INTERACTIVE VISUAL CHECK
# ---------------------------------------------------------------------------
print("[INFO] Launching interactive plot …  (close window to exit)")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.15, "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

x_max = max(t[-1] for t in TIME_ALL)
y_max = max(t.max() for t in TOT_RAW_ALL)

raw_line,  = ax.plot(TIME_ALL[0], TOT_RAW_ALL[0],
                      color="0.6", lw=1, label="ΣP raw")
filt_line, = ax.plot(TIME_ALL[0], TOT_FILT_ALL[0],
                      color="0.2", lw=0.8, label="LPF ΣP")
vt0 = ax.axvline(TIME_ALL[0][IDX_T0_ALL[0]],
                 color="C0", ls="--", label="t₀")
vpk = ax.axvline(TIME_ALL[0][IDX_PK_ALL[0]],
                 color="C3", ls="--", label="t_peak")

ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max * 1.05)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Total Pressure [Pa]")
ax.legend(loc="upper right")

title = ax.set_title("", fontsize=12)

# Slider --------------------------------------------------------------------
ax_idx = plt.axes([0.12, 0.1, 0.76, 0.04])
slider = Slider(ax_idx, "Sample", 0, len(TIME_ALL) - 1,
                valinit=0, valstep=1)

def _update(idx):
    idx = int(idx)

    raw_line.set_data(TIME_ALL[idx], TOT_RAW_ALL[idx])
    filt_line.set_data(TIME_ALL[idx], TOT_FILT_ALL[idx])

    x_t0 = TIME_ALL[idx][IDX_T0_ALL[idx]]
    x_pk = TIME_ALL[idx][IDX_PK_ALL[idx]]

    # Matplotlib Line2D expects sequences, not scalars
    vt0.set_xdata([x_t0, x_t0])
    vpk.set_xdata([x_pk, x_pk])

    ax.set_ylim(0, max(TOT_RAW_ALL[idx].max(), y_max) * 1.05)

    title.set_text(
        f"Sample {idx}  |  Rn={all_Rn[idx]:.3f}  "
        f"dr={all_dr[idx]:.2f} m  H={all_H[idx]:.2f} m  "
        f"Δt={dT_list[idx]:.3f} s")
    fig.canvas.draw_idle()

slider.on_changed(_update)
_update(0)
plt.show()

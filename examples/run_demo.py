# -*- coding: utf-8 -*-
"""Create a tiny synthetic Excel file and run a quick PILOT smoke experiment."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
demo_dir = ROOT / "outputs" / "demo"
demo_dir.mkdir(parents=True, exist_ok=True)
xlsx = demo_dir / "demo_telemetry.xlsx"

rng = np.random.default_rng(42)
T, N = 120, 3
t = np.arange(T)
data = {"time": t}
for i in range(N):
    data[f"sensor_{i+1}"] = np.sin(2 * np.pi * t / (24 + 3 * i)) + 0.02 * t / T + 0.05 * rng.standard_normal(T)
# inject a small anomaly-like segment
for i in [1]:
    data[f"sensor_{i+1}"][60:68] += 1.2
pd.DataFrame(data).to_excel(xlsx, index=False)

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_train_test.py"),
    "--excel_path", str(xlsx),
    "--output_dir", str(demo_dir / "results"),
    "--epochs", "1",
    "--batch_size", "32",
    "--seq_len", "8",
    "--pred_len", "2",
    "--models", "PILOT-Temporal",
    "--cpu",
]
raise SystemExit(subprocess.call(cmd))

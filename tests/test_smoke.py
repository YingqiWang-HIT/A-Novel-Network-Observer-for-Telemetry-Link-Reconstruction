# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot_recovery.models.pilot import PILOTObserver


def test_pilot_forward_shape():
    B, L, N, H = 2, 16, 5, 3
    input_dim = 3 * N + 9
    A = np.eye(N, dtype=np.float32)
    model = PILOTObserver(A, A, A, A, input_dim=input_dim, seq_len=L, pred_len=H, n_channels=N)
    x = torch.randn(B, L, input_dim)
    x[:, :, N:2*N] = torch.sigmoid(x[:, :, N:2*N])
    x[:, :, 2*N:3*N] = torch.sigmoid(x[:, :, 2*N:3*N])
    y, extra = model(x)
    assert y.shape == (B, H, N)
    assert "uncertainty" in extra

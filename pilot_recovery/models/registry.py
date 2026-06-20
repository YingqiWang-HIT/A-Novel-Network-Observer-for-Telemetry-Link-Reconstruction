# -*- coding: utf-8 -*-
"""Model registry."""
from __future__ import annotations

from typing import Dict
import torch.nn as nn

from .pilot import PILOTObserver
from ..config import Config


def build_models(data: dict, cfg: Config) -> Dict[str, nn.Module]:
    """Build the proposed PILOT model and ablation variants.

    Official third-party baseline implementations are intentionally not bundled in this
    public repository. See ``baselines/`` for adapter placeholders and README notes.
    """
    kwargs = dict(
        A=data["A"],
        A_local=data["A_local"],
        A_cross=data["A_cross"],
        A_global=data["A_global"],
        input_dim=data["input_dim"],
        seq_len=data["seq_len"],
        pred_len=data["pred_len"],
        n_channels=data["N"],
        lags=cfg.delay_lags,
    )
    models: Dict[str, nn.Module] = {
        "PILOT-Temporal": PILOTObserver(**kwargs, mode="temporal", state_dim=48, hidden=128, dropout=0.08),
        "PILOT-w/o-Graph": PILOTObserver(**kwargs, mode="no_graph", state_dim=48, hidden=128, dropout=0.08),
        "PILOT-w/o-Separator": PILOTObserver(**kwargs, mode="no_separator", state_dim=48, hidden=128, dropout=0.08),
        "PILOT-Full": PILOTObserver(**kwargs, mode="full", state_dim=48, hidden=128, dropout=0.08),
    }
    if cfg.run_model_names is not None:
        keep = set(cfg.run_model_names)
        models = {name: model for name, model in models.items() if name in keep}
        missing = keep.difference(models.keys())
        if missing:
            raise ValueError(f"Unknown model(s): {sorted(missing)}. Available: {list(build_models(data, Config()).keys())}")
    return models

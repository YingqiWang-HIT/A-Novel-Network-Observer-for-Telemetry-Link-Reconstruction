# -*- coding: utf-8 -*-
"""Metrics and prediction utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .config import Config
from .losses import unpack_output


def denorm(y_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return y_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _macro_channel_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None, cfg: Config) -> Dict[str, float]:
    if mask is not None:
        denom = mask.sum(axis=(0, 1))
        valid = denom > 0
        if not np.any(valid):
            return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        err = pred - true
        mse_ch = ((err ** 2) * mask).sum(axis=(0, 1))[valid] / (denom[valid] + 1e-8)
        mae_ch = (np.abs(err) * mask).sum(axis=(0, 1))[valid] / (denom[valid] + 1e-8)
        eps_ch = cfg.mape_eps_ratio * np.maximum(np.abs(true).mean(axis=(0, 1))[valid], 1e-6)
        mape_ch = ((np.abs(err) / (np.abs(true) + eps_ch.reshape(1, 1, -1))) * mask[:, :, valid]).sum(axis=(0, 1)) / (denom[valid] + 1e-8)
    else:
        err = pred - true
        mse_ch = (err ** 2).mean(axis=(0, 1))
        mae_ch = np.abs(err).mean(axis=(0, 1))
        eps_ch = cfg.mape_eps_ratio * np.maximum(np.abs(true).mean(axis=(0, 1)), 1e-6)
        mape_ch = (np.abs(err) / (np.abs(true) + eps_ch.reshape(1, 1, -1))).mean(axis=(0, 1))
    return {"RMSE": float(np.sqrt(np.mean(mse_ch))), "MAE": float(np.mean(mae_ch)), "MAPE": float(np.mean(mape_ch) * 100.0)}


def compute_metrics(pred_norm: np.ndarray, true_norm: np.ndarray, miss_mask: np.ndarray, mean: np.ndarray, std: np.ndarray, cfg: Config) -> Dict:
    pred = denorm(pred_norm, mean, std)
    true = denorm(true_norm, mean, std)
    all_metrics = _macro_channel_metrics(pred, true, None, cfg)
    missing_metrics = _macro_channel_metrics(pred, true, miss_mask, cfg)
    per_channel_rmse = np.sqrt(((pred - true) ** 2).mean(axis=(0, 1)))
    return {
        "All_RMSE": all_metrics["RMSE"],
        "All_MAE": all_metrics["MAE"],
        "All_MAPE": all_metrics["MAPE"],
        "Missing_RMSE": missing_metrics["RMSE"],
        "Missing_MAE": missing_metrics["MAE"],
        "Missing_MAPE": missing_metrics["MAPE"],
        "Per_Channel_RMSE": per_channel_rmse,
        "pred_norm": pred_norm,
        "true_norm": true_norm,
        "miss_mask": miss_mask,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, mean, std, cfg: Config) -> Dict:
    model.eval()
    preds, trues, masks = [], [], []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        pred, _ = unpack_output(out)
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.numpy())
        masks.append(mb.numpy())
    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    miss_mask = np.concatenate(masks, axis=0)
    return compute_metrics(pred_norm, true_norm, miss_mask, mean, std, cfg)

# -*- coding: utf-8 -*-
"""Training losses for PILOT."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import Config


def unpack_output(out):
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def make_channel_weight(std, device, clip=(0.25, 6.0)):
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    inv = 1.0 / (std_t + 1e-6)
    inv = inv / inv.mean().clamp_min(1e-6)
    return torch.clamp(inv, clip[0], clip[1])


def standard_selection_loss(pred: torch.Tensor, target: torch.Tensor, diff_weight: float = 0.05) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if pred.size(1) > 1:
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_true = target[:, 1:] - target[:, :-1]
        mse = mse + diff_weight * F.mse_loss(d_pred, d_true)
    return mse


def forecasting_loss(
    out,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    model_name: str,
    epoch: int,
    cfg: Config,
    channel_weight: torch.Tensor | None = None,
    A_torch: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite loss.

    ``missing_mask`` is 1 on artificially removed target positions. PILOT auxiliary terms
    are active after ``cfg.aux_start_epoch`` so that early training first learns stable
    reconstruction.
    """
    pred, extra = unpack_output(out)
    if channel_weight is None:
        w = 1.0
    else:
        w = channel_weight.view(1, 1, -1)

    err = pred - target
    weighted_err = err * w
    mse = (weighted_err ** 2).mean()
    huber = F.smooth_l1_loss(weighted_err, torch.zeros_like(weighted_err), beta=0.5)
    loss = 0.55 * huber + 0.45 * mse
    log = {"mse": float(mse.detach().cpu()), "huber": float(huber.detach().cpu())}

    if pred.size(1) > 1:
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_true = target[:, 1:] - target[:, :-1]
        diff = F.smooth_l1_loss((d_pred - d_true) * w, torch.zeros_like(d_pred), beta=0.5)
        loss = loss + cfg.diff_loss_weight * diff
        log["diff"] = float(diff.detach().cpu())

    if missing_mask.sum() > 0:
        miss_mse = (((weighted_err ** 2) * missing_mask).sum() / (missing_mask.sum() + 1e-6))
        loss = loss + cfg.missing_loss_weight * miss_mse
        log["missing"] = float(miss_mse.detach().cpu())

    if epoch >= cfg.aux_start_epoch and isinstance(extra, dict):
        support_prob = extra.get("support_prob")
        if support_prob is not None:
            # Weak pseudo-target: a channel that has future artificial deletion or strong future variation
            # should keep higher anomaly/recovery evidence.
            future_miss = (missing_mask.max(dim=1).values > 0.5).float()
            future_amp = target.std(dim=1)
            amp_thr = future_amp.detach().mean(dim=1, keepdim=True) + future_amp.detach().std(dim=1, keepdim=True)
            amp_flag = (future_amp > amp_thr).float()
            support_target = torch.clamp(future_miss + amp_flag, 0, 1)
            support = F.binary_cross_entropy(support_prob.clamp(1e-4, 1 - 1e-4), support_target)
            loss = loss + cfg.support_loss_weight * support
            log["support"] = float(support.detach().cpu())

        uncertainty = extra.get("uncertainty")
        if uncertainty is not None:
            var = uncertainty.clamp_min(1e-5)
            nll = 0.5 * ((err.detach() ** 2) / var + torch.log(var)).mean()
            loss = loss + cfg.uncertainty_loss_weight * nll
            log["uncertainty_nll"] = float(nll.detach().cpu())

        projected = extra.get("projected_anomaly")
        if projected is not None:
            sparse = projected.abs().mean()
            persist = torch.tensor(0.0, device=projected.device)
            if projected.size(1) > 1:
                persist = (projected[:, 1:] - projected[:, :-1]).abs().mean()
            loss = loss + cfg.anomaly_sparse_weight * sparse + cfg.anomaly_persist_weight * persist
            log["anomaly_sparse"] = float(sparse.detach().cpu())
            log["anomaly_persist"] = float(persist.detach().cpu())

    log["total"] = float(loss.detach().cpu())
    return loss, log

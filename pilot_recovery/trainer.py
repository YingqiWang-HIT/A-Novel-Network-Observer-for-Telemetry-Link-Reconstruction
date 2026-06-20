# -*- coding: utf-8 -*-
"""Training and evaluation loop."""
from __future__ import annotations

import copy
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .losses import forecasting_loss, make_channel_weight, standard_selection_loss, unpack_output
from .metrics import evaluate_model
from .utils import count_parameters, sync_device


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k in state:
                state[k].copy_(v)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, model_name: str, device: torch.device, cfg: Config, channel_weight=None, A_torch=None) -> float:
    model.eval()
    losses = []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        out = model(xb)
        pred, _ = unpack_output(out)
        if cfg.val_select_standard:
            loss = standard_selection_loss(pred, yb)
        else:
            loss, _ = forecasting_loss(out, yb, mb, model_name, epoch=10**9, cfg=cfg, channel_weight=channel_weight, A_torch=A_torch)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: Config,
    std,
    A_np,
) -> Tuple[nn.Module, Dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    channel_weight = make_channel_weight(std, device)
    A_torch = torch.tensor(A_np, dtype=torch.float32, device=device)
    ema = EMA(model, decay=0.995)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    bad_count = 0
    history = []
    t0 = time.time()

    print(f"\n[Train] {model_name}: params={count_parameters(model):,}, epochs={cfg.epochs}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                out = model(xb)
                loss, _ = forecasting_loss(out, yb, mb, model_name, epoch, cfg, channel_weight=channel_weight, A_torch=A_torch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_loss = evaluate_loss(model, val_loader, model_name, device, cfg, channel_weight=channel_weight, A_torch=A_torch)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_count = 0
        else:
            bad_count += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"  epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")
        if bad_count >= cfg.patience:
            print(f"  early stop at epoch={epoch}, best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    # Keep a stable copy of the best state. EMA is not blindly copied because validation selected raw state.
    sync_device(device)
    info = {
        "model_name": model_name,
        "params": count_parameters(model),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": time.time() - t0,
        "history": history,
    }
    return model, info


def train_and_evaluate(models: Dict[str, nn.Module], loaders, data: Dict, cfg: Config, device: torch.device):
    train_loader, val_loader, test_loader = loaders
    results = {}
    train_infos = {}
    failed = []
    for name, model in models.items():
        try:
            trained, info = train_model(name, model, train_loader, val_loader, device, cfg, data["std"], data["A"])
            res = evaluate_model(trained, test_loader, device, data["mean"], data["std"], cfg)
            results[name] = res
            train_infos[name] = info
            print(f"[Test] {name}: All_RMSE={res['All_RMSE']:.6f}, Missing_RMSE={res['Missing_RMSE']:.6f}")
        except Exception as exc:
            failed.append({"model": name, "error": repr(exc)})
            print(f"[Error] {name}: {exc}")
            if not cfg.continue_on_error:
                raise
    return results, train_infos, failed

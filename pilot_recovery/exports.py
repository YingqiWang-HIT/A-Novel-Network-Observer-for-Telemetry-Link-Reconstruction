# -*- coding: utf-8 -*-
"""Result exporting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import Config
from .utils import ensure_dir, sanitize_filename, save_json, json_safe
from .metrics import denorm


def save_dataset_details(data: Dict, cfg: Config, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    details = {
        "T": data["T"],
        "N": data["N"],
        "input_dim": data["input_dim"],
        "seq_len": data["seq_len"],
        "pred_len": data["pred_len"],
        "channels": data["channel_names"],
        "subsystems": data["subsystems"],
        "train_samples": len(data["X_train"]),
        "val_samples": len(data["X_val"]),
        "test_samples": len(data["X_test"]),
    }
    save_json(out_dir / "dataset_details.json", details)
    if cfg.save_adjacency_matrix:
        pd.DataFrame(data["A"], index=data["channel_names"], columns=data["channel_names"]).to_csv(out_dir / "adjacency_global.csv", encoding="utf-8-sig")
        pd.DataFrame(data["A_local"], index=data["channel_names"], columns=data["channel_names"]).to_csv(out_dir / "adjacency_local.csv", encoding="utf-8-sig")
        pd.DataFrame(data["A_cross"], index=data["channel_names"], columns=data["channel_names"]).to_csv(out_dir / "adjacency_cross.csv", encoding="utf-8-sig")


def save_training_history(model_name: str, info: Dict, cfg: Config, out_dir: str | Path) -> None:
    if not cfg.save_training_history:
        return
    hist_dir = ensure_dir(Path(out_dir) / cfg.training_history_dir_name)
    hist = pd.DataFrame(info.get("history", []))
    if len(hist):
        hist.to_csv(hist_dir / f"{sanitize_filename(model_name)}_history.csv", index=False, encoding="utf-8-sig")


def save_model_artifact(model_name: str, model: torch.nn.Module, info: Dict, cfg: Config, out_dir: str | Path) -> None:
    if not cfg.save_trained_models:
        return
    model_dir = ensure_dir(Path(out_dir) / cfg.models_dir_name)
    payload = {
        "model_name": model_name,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "info": json_safe(info),
    }
    torch.save(payload, model_dir / f"{sanitize_filename(model_name)}.pt")


def save_metrics(results: Dict, train_infos: Dict, channel_names: List[str], out_dir: str | Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    rows = []
    per_channel = []
    for name, res in results.items():
        info = train_infos.get(name, {})
        rows.append({
            "Model": name,
            "All_RMSE": res.get("All_RMSE"),
            "All_MAE": res.get("All_MAE"),
            "All_MAPE(%)": res.get("All_MAPE"),
            "Missing_RMSE": res.get("Missing_RMSE"),
            "Missing_MAE": res.get("Missing_MAE"),
            "Missing_MAPE(%)": res.get("Missing_MAPE"),
            "Params": info.get("params"),
            "Best_Val": info.get("best_val"),
            "Best_Epoch": info.get("best_epoch"),
            "Train_Seconds": info.get("train_seconds"),
        })
        rmse = res.get("Per_Channel_RMSE")
        if rmse is not None:
            for ch, val in zip(channel_names, rmse):
                per_channel.append({"Model": name, "Channel": ch, "RMSE": float(val)})
    df = pd.DataFrame(rows).sort_values("Missing_RMSE", na_position="last")
    df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_excel(out_dir / "metrics_summary.xlsx", index=False)
    except Exception:
        pass
    if per_channel:
        pc = pd.DataFrame(per_channel)
        pc.to_csv(out_dir / "per_channel_rmse.csv", index=False, encoding="utf-8-sig")
        try:
            pc.to_excel(out_dir / "per_channel_rmse.xlsx", index=False)
        except Exception:
            pass
    return df


def export_predictions_wide(results: Dict, channel_names: List[str], mean, std, cfg: Config, out_dir: str | Path) -> None:
    if not (cfg.save_prediction_csv or cfg.save_prediction_excel):
        return
    pred_dir = ensure_dir(Path(out_dir) / "predictions")
    for name, res in results.items():
        pred = denorm(res["pred_norm"], mean, std)
        true = denorm(res["true_norm"], mean, std)
        B, H, N = pred.shape
        rows = []
        for i in range(B):
            row = {"sample": i}
            for h in range(H):
                for j, ch in enumerate(channel_names):
                    row[f"pred_h{h+1}_{ch}"] = pred[i, h, j]
                    row[f"true_h{h+1}_{ch}"] = true[i, h, j]
            rows.append(row)
        df = pd.DataFrame(rows)
        safe = sanitize_filename(name)
        if cfg.save_prediction_csv:
            df.to_csv(pred_dir / f"{safe}_predictions.csv", index=False, encoding="utf-8-sig")
        if cfg.save_prediction_excel and df.shape[0] * df.shape[1] <= cfg.max_excel_cells:
            try:
                df.to_excel(pred_dir / f"{safe}_predictions.xlsx", index=False)
            except Exception:
                pass


def save_output_manifest(cfg: Config, data: Dict, metrics_df: Optional[pd.DataFrame], failed_models: Optional[List], out_dir: str | Path) -> None:
    if not cfg.save_output_manifest:
        return
    manifest = {
        "config": cfg.__dict__,
        "dataset": {"T": data["T"], "N": data["N"], "seq_len": data["seq_len"], "pred_len": data["pred_len"]},
        "metrics": metrics_df.to_dict(orient="records") if metrics_df is not None else None,
        "failed_models": failed_models or [],
    }
    save_json(Path(out_dir) / "output_manifest.json", manifest)

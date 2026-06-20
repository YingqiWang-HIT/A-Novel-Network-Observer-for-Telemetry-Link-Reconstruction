# -*- coding: utf-8 -*-
"""Train and test PILOT on a single Excel file or a folder of Excel files."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot_recovery.config import load_config, Config
from pilot_recovery.data import discover_dataset_files, make_loaders, prepare_dataset
from pilot_recovery.exports import (
    export_predictions_wide,
    save_dataset_details,
    save_metrics,
    save_model_artifact,
    save_output_manifest,
    save_training_history,
)
from pilot_recovery.models import build_models
from pilot_recovery.trainer import train_and_evaluate
from pilot_recovery.utils import ensure_dir, get_device, sanitize_filename, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="PILOT telemetry recovery training/testing")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument("--excel_path", type=str, default=None, help="Single Excel dataset path")
    parser.add_argument("--dataset_folder", type=str, default=None, help="Folder containing Excel datasets")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names, e.g. PILOT-Full,PILOT-w/o-Graph")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def apply_cli_overrides(args) -> Config:
    overrides = {
        "excel_path": args.excel_path,
        "dataset_folder": args.dataset_folder,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
    }
    cfg = load_config(args.config, overrides=overrides)
    if args.models:
        cfg.run_model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.cpu:
        cfg.use_gpu = False
    return cfg


def run_one_dataset(base_cfg: Config, excel_path: str, root_out: Path, device) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg.excel_path = excel_path
    dataset_name = sanitize_filename(Path(excel_path).stem)
    out_dir = ensure_dir(root_out / dataset_name)
    print(f"\n========== Dataset: {excel_path} ==========")
    print(f"[Output] {out_dir}")

    set_seed(cfg.seed)
    data = prepare_dataset(cfg)
    if cfg.save_dataset_details:
        save_dataset_details(data, cfg, out_dir)
    loaders = make_loaders(data, cfg)
    models = build_models(data, cfg)
    results, train_infos, failed = train_and_evaluate(models, loaders, data, cfg, device)

    for name, model in models.items():
        if name in train_infos:
            save_training_history(name, train_infos[name], cfg, out_dir)
            save_model_artifact(name, model, train_infos[name], cfg, out_dir)
    metrics_df = save_metrics(results, train_infos, data["channel_names"], out_dir)
    export_predictions_wide(results, data["channel_names"], data["mean"], data["std"], cfg, out_dir)
    save_output_manifest(cfg, data, metrics_df, failed, out_dir)


def main():
    args = parse_args()
    cfg = apply_cli_overrides(args)
    root_out = ensure_dir(cfg.output_dir)
    device = get_device(cfg.use_gpu)
    files = discover_dataset_files(cfg)
    if not files:
        raise FileNotFoundError("No Excel datasets were found.")
    print(f"[Run] Found {len(files)} dataset(s).")
    for path in files:
        run_one_dataset(cfg, path, root_out, device)


if __name__ == "__main__":
    main()

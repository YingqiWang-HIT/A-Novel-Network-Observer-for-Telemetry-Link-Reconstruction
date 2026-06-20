# -*- coding: utf-8 -*-
"""Configuration objects for PILOT telemetry recovery experiments."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict


@dataclass
class Config:
    # Data
    excel_path: str = "data/example.xlsx"
    sheet_name: int | str = 0
    time_col: Optional[str] = None
    first_col_is_time: bool = True
    output_dir: str = "outputs/pilot_results"

    # Folder mode
    dataset_folder: Optional[str] = None
    recursive_scan: bool = False
    dataset_patterns: Tuple[str, ...] = ("*.xlsx", "*.xls")
    skip_excel_temp_files: bool = True
    skip_result_excel_files: bool = True

    # Forecasting window
    seq_len: int = 96
    pred_len: int = 12
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    auto_adjust_window: bool = True
    min_seq_len: int = 24
    min_pred_len: int = 1
    min_total_windows: int = 80
    skip_too_short_dataset: bool = True

    # Artificial missingness
    seed: int = 42
    point_missing_rate: float = 0.05
    block_missing_rate: float = 0.025
    num_channel_blocks: int = 25
    channel_block_min_len: int = 4
    channel_block_max_len: int = 24
    num_system_blocks: int = 8
    system_block_min_len: int = 4
    system_block_max_len: int = 20

    # Graph construction
    top_k_graph: int = 6
    graph_threshold: float = 0.05
    subsystem_count: int = 4
    delay_lags: Tuple[int, ...] = (0, 1, 3, 6, 12)

    # Metrics
    mape_eps_ratio: float = 0.01

    # Training
    use_gpu: bool = True
    batch_size: int = 128
    epochs: int = 260
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 45
    min_delta: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 0

    # PILOT loss scheduling
    aux_start_epoch: int = 40
    missing_loss_weight: float = 0.10
    diff_loss_weight: float = 0.10
    scale_loss_weight: float = 0.05
    support_loss_weight: float = 0.02
    uncertainty_loss_weight: float = 0.01
    anomaly_sparse_weight: float = 1e-4
    anomaly_persist_weight: float = 5e-5
    val_select_standard: bool = True

    # Model selection
    run_model_names: Optional[List[str]] = None
    continue_on_error: bool = False

    # Export
    save_prediction_excel: bool = False
    save_prediction_csv: bool = True
    max_excel_cells: int = 8_000_000
    plot_channels: int = 4
    plot_len: int = 450
    save_trained_models: bool = True
    models_dir_name: str = "trained_models"
    save_training_history: bool = True
    training_history_dir_name: str = "training_history"
    save_dataset_details: bool = True
    save_adjacency_matrix: bool = True
    save_output_manifest: bool = True


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    valid = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key in valid:
            current = getattr(obj, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    cfg = Config()
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError("Please install PyYAML or use a JSON config file.") from exc
            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        elif path.suffix.lower() == ".json":
            import json
            values = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Only .yaml, .yml, and .json config files are supported.")
        _update_dataclass(cfg, values)
    if overrides:
        _update_dataclass(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg

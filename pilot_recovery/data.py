# -*- coding: utf-8 -*-
"""Excel telemetry dataset preparation for PILOT."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


def read_excel_time_series(
    excel_path: str,
    sheet_name: int | str = 0,
    time_col: Optional[str] = None,
    first_col_is_time: bool = True,
):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError(f"Excel file must contain at least a time column and one sensor column. Got shape={df.shape}")

    time_values = None
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col={time_col!r} not found in columns: {list(df.columns)[:10]}")
        time_values = df[time_col].values
        df = df.drop(columns=[time_col])
        print(f"[Data] Dropped specified time column: {time_col}")
    elif first_col_is_time:
        first_col = df.columns[0]
        time_values = df.iloc[:, 0].values
        df = df.iloc[:, 1:].copy()
        print(f"[Data] Dropped first column as time: {first_col}")
    else:
        for col in list(df.columns):
            col_name = str(col).strip().lower()
            if col_name in {"time", "timestamp", "date", "datetime", "index", "t"} or "time" in col_name:
                time_values = df[col].values
                df = df.drop(columns=[col])
                print(f"[Data] Auto-dropped time/index column: {col}")
                break

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 1:
        raise ValueError("No sensor columns remain after removing time columns.")

    channel_names = [str(c) for c in df.columns]
    data_raw = df.values.astype(np.float32)
    print(f"[Data] Loaded {excel_path}; shape={data_raw.shape}; channels={len(channel_names)}")
    return data_raw, channel_names, time_values


def interpolate_target(data_raw: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data_raw)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill().fillna(0.0)
    return df.values.astype(np.float32)


def forward_fill_observation(data_with_nan: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data_with_nan)
    df = df.ffill().bfill().fillna(0.0)
    return df.values.astype(np.float32)


def create_artificial_missing_mask(T: int, N: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Return artificial_observed_mask: 1 = observed, 0 = artificially removed."""
    mask = np.ones((T, N), dtype=np.float32)
    point = rng.random((T, N)) < cfg.point_missing_rate
    mask[point] = 0.0

    for _ in range(cfg.num_channel_blocks):
        ch = int(rng.integers(0, N))
        length = int(rng.integers(cfg.channel_block_min_len, cfg.channel_block_max_len + 1))
        start = int(rng.integers(0, max(1, T - length)))
        mask[start:start + length, ch] = 0.0

    group_count = min(4, max(1, N // 5))
    groups = np.array_split(np.arange(N), group_count)
    for _ in range(cfg.num_system_blocks):
        group = groups[int(rng.integers(0, len(groups)))]
        length = int(rng.integers(cfg.system_block_min_len, cfg.system_block_max_len + 1))
        start = int(rng.integers(0, max(1, T - length)))
        mask[start:start + length, group] = 0.0

    return mask.astype(np.float32)


def compute_delta(mask: np.ndarray, clip_value: int = 80) -> np.ndarray:
    T, N = mask.shape
    delta = np.zeros((T, N), dtype=np.float32)
    for t in range(1, T):
        delta[t] = np.where(mask[t] > 0.5, 0.0, delta[t - 1] + 1.0)
    return (np.clip(delta, 0, clip_value) / float(clip_value)).astype(np.float32)


def add_time_features(T: int, periods=(24, 60, 180, 360)) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    feats = []
    for p in periods:
        feats.append(np.sin(2 * np.pi * t / p))
        feats.append(np.cos(2 * np.pi * t / p))
    feats.append(t / max(1, T - 1))
    return np.stack(feats, axis=-1).astype(np.float32)


def build_correlation_adjacency(target_norm: np.ndarray, train_end: int, top_k: int = 6, threshold: float = 0.05) -> np.ndarray:
    train = target_norm[:train_end]
    corr = np.corrcoef(train.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 1.0)

    N = corr.shape[0]
    k = min(top_k, N)
    A = np.zeros_like(corr, dtype=np.float32)
    for i in range(N):
        idx = np.argsort(corr[i])[-k:]
        A[i, idx] = corr[i, idx]
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)
    A = np.maximum(A, A.T)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A.astype(np.float32)


def build_hierarchical_adjacency(A: np.ndarray, n_subsystems: int):
    N = A.shape[0]
    groups = np.array_split(np.arange(N), max(1, min(n_subsystems, N)))

    A_local = np.eye(N, dtype=np.float32)
    for g in groups:
        A_local[np.ix_(g, g)] = A[np.ix_(g, g)]

    A_cross = A.copy()
    for g in groups:
        A_cross[np.ix_(g, g)] = 0.0
    np.fill_diagonal(A_cross, 1.0)
    A_global = A.copy()

    def normalize(M: np.ndarray):
        M = np.maximum(M, M.T)
        np.fill_diagonal(M, 1.0)
        return (M / (M.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    return normalize(A_local), normalize(A_cross), normalize(A_global), [list(map(int, g)) for g in groups]


def make_windows(features: np.ndarray, target: np.ndarray, artificial_missing: np.ndarray, seq_len: int, pred_len: int):
    X, Y, M = [], [], []
    max_i = len(features) - seq_len - pred_len + 1
    for i in range(max_i):
        X.append(features[i:i + seq_len])
        Y.append(target[i + seq_len:i + seq_len + pred_len])
        M.append(1.0 - artificial_missing[i + seq_len:i + seq_len + pred_len])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), np.asarray(M, dtype=np.float32)


def _auto_adjust_window(cfg: Config, T: int) -> None:
    required = cfg.seq_len + cfg.pred_len + cfg.min_total_windows
    if T >= required:
        return
    old_seq, old_pred = cfg.seq_len, cfg.pred_len
    new_pred = min(cfg.pred_len, max(cfg.min_pred_len, max(1, T // 20)))
    max_seq = T - new_pred - cfg.min_total_windows
    if max_seq < cfg.min_seq_len:
        max_seq = T - new_pred - 20
    new_seq = min(cfg.seq_len, max(4, max_seq))
    if new_seq < 4 or T - new_seq - new_pred + 1 < 20:
        msg = f"Dataset too short for windowing: T={T}, candidate seq_len={new_seq}, pred_len={new_pred}."
        if cfg.skip_too_short_dataset:
            raise ValueError(msg)
        raise ValueError(msg)
    cfg.seq_len = int(new_seq)
    cfg.pred_len = int(new_pred)
    print(f"[Data] Auto-adjust window: seq_len {old_seq}->{cfg.seq_len}, pred_len {old_pred}->{cfg.pred_len}")


def prepare_dataset(cfg: Config) -> Dict:
    data_raw, channel_names, time_values = read_excel_time_series(
        cfg.excel_path, cfg.sheet_name, cfg.time_col, first_col_is_time=cfg.first_col_is_time
    )
    target_data = interpolate_target(data_raw)
    T, N = target_data.shape
    if cfg.auto_adjust_window:
        _auto_adjust_window(cfg, T)

    rng = np.random.default_rng(cfg.seed)
    natural_observed = (~np.isnan(data_raw)).astype(np.float32)
    artificial_observed = create_artificial_missing_mask(T, N, cfg, rng)
    final_observed = natural_observed * artificial_observed

    obs_data = target_data.copy()
    obs_data[final_observed < 0.5] = np.nan
    obs_filled = forward_fill_observation(obs_data)

    train_time_end = int(T * cfg.train_ratio)
    train_target = target_data[:train_time_end]
    mean = train_target.mean(axis=0).astype(np.float32)
    std = train_target.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    target_norm = ((target_data - mean) / std).astype(np.float32)
    obs_norm = ((obs_filled - mean) / std).astype(np.float32)
    delta = compute_delta(final_observed)
    time_feat = add_time_features(T)
    features = np.concatenate([obs_norm, final_observed, delta, time_feat], axis=-1).astype(np.float32)

    A = build_correlation_adjacency(target_norm, train_time_end, cfg.top_k_graph, cfg.graph_threshold)
    A_local, A_cross, A_global, subsystems = build_hierarchical_adjacency(A, cfg.subsystem_count)
    X_all, Y_all, M_all = make_windows(features, target_norm, artificial_observed, cfg.seq_len, cfg.pred_len)
    n_samples = len(X_all)
    if n_samples <= 0:
        raise ValueError("Time series too short. Reduce seq_len or pred_len.")

    n_train = int(n_samples * cfg.train_ratio)
    n_val = int(n_samples * cfg.val_ratio)
    if n_samples >= 3:
        n_train = min(max(1, n_train), n_samples - 2)
        n_val = min(max(1, n_val), n_samples - n_train - 1)
    if n_train <= 0 or n_val <= 0 or n_samples - n_train - n_val <= 0:
        raise ValueError(f"Not enough samples: total={n_samples}, train={n_train}, val={n_val}")

    data = {
        "X_train": torch.tensor(X_all[:n_train]),
        "Y_train": torch.tensor(Y_all[:n_train]),
        "M_train": torch.tensor(M_all[:n_train]),
        "X_val": torch.tensor(X_all[n_train:n_train + n_val]),
        "Y_val": torch.tensor(Y_all[n_train:n_train + n_val]),
        "M_val": torch.tensor(M_all[n_train:n_train + n_val]),
        "X_test": torch.tensor(X_all[n_train + n_val:]),
        "Y_test": torch.tensor(Y_all[n_train + n_val:]),
        "M_test": torch.tensor(M_all[n_train + n_val:]),
        "mean": mean,
        "std": std,
        "A": A,
        "A_local": A_local,
        "A_cross": A_cross,
        "A_global": A_global,
        "subsystems": subsystems,
        "channel_names": channel_names,
        "N": N,
        "T": T,
        "input_dim": features.shape[-1],
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "time_values": time_values,
    }
    print(f"[Data] Samples train/val/test: {len(data['X_train'])}/{len(data['X_val'])}/{len(data['X_test'])}")
    print(f"[Data] Artificial missing rate: {float(1.0 - artificial_observed.mean()):.4f}")
    return data


def make_loaders(data: Dict, cfg: Config):
    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["Y_train"], data["M_train"]),
        batch_size=cfg.batch_size, shuffle=True, pin_memory=(cfg.use_gpu and torch.cuda.is_available()), num_workers=cfg.num_workers, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(data["X_val"], data["Y_val"], data["M_val"]),
        batch_size=cfg.batch_size, shuffle=False, pin_memory=(cfg.use_gpu and torch.cuda.is_available()), num_workers=cfg.num_workers, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(data["X_test"], data["Y_test"], data["M_test"]),
        batch_size=cfg.batch_size, shuffle=False, pin_memory=(cfg.use_gpu and torch.cuda.is_available()), num_workers=cfg.num_workers, drop_last=False
    )
    return train_loader, val_loader, test_loader


def discover_dataset_files(cfg: Config) -> List[str]:
    if not cfg.dataset_folder:
        return [cfg.excel_path]
    root = Path(cfg.dataset_folder)
    if not root.exists():
        raise FileNotFoundError(f"dataset_folder does not exist: {root}")
    files: List[Path] = []
    for pattern in cfg.dataset_patterns:
        files.extend(root.rglob(pattern) if cfg.recursive_scan else root.glob(pattern))
    unique = []
    seen = set()
    for path in sorted(files):
        if cfg.skip_excel_temp_files and path.name.startswith("~$"):
            continue
        if cfg.skip_result_excel_files and any(k in path.name.lower() for k in ["metric", "result", "prediction", "summary"]):
            continue
        if path.resolve() not in seen:
            seen.add(path.resolve())
            unique.append(str(path))
    return unique

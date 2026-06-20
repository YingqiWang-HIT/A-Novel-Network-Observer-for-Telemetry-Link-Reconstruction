# PILOT: Anomaly-Retentive Spacecraft Telemetry Recovery

This repository provides the public PyTorch implementation of **A-Novel-Network-Observer-for-Telemetry-Link-Reconstruction**.

PILOT reconstructs intermittent multivariate spacecraft telemetry under irregular link loss while preserving anomaly evidence. The released code focuses on the proposed method and its ablation variants:

- `PILOT-Full`: full innovation-separable hybrid networked observer;
- `PILOT-Temporal`: event-time mixed-spectrum observer without anomaly correction;
- `PILOT-w/o-Graph`: removes reliability-conditioned hypergraph coupling;
- `PILOT-w/o-Separator`: removes finite-window projected anomaly separation.

## Repository structure

```text
PILOT-Telemetry-Recovery/
├── pilot_recovery/              # Main Python package
│   ├── config.py                # Experiment configuration dataclass
│   ├── data.py                  # Excel loading, missing-mask construction, graph building
│   ├── losses.py                # PILOT training losses
│   ├── metrics.py               # RMSE/MAE/MAPE and prediction utilities
│   ├── trainer.py               # Training and validation loops
│   ├── exports.py               # Metrics, checkpoints, manifest export
│   └── models/
│       ├── pilot.py             # PILOT model modules
│       ├── common.py            # Shared layers
│       └── registry.py          # Model/ablation registry
├── baselines/                   # One adapter file per comparison method
│   ├── kalman_smoother.py
│   ├── saits.py
│   ├── grin.py
│   ├── spin.py
│   ├── csdi.py
│   ├── bayotide.py
│   ├── mamba.py
│   ├── linoss.py
│   ├── timba.py
│   ├── kedgn.py
│   ├── hyperimts.py
│   └── imdiffusion.py
├── configs/                     # YAML configs
├── scripts/                     # CLI entry points
├── docs/                        # Reproducibility and baseline notices
├── examples/                    # Synthetic demo
├── tests/                       # Smoke tests
└── data/                        # Local data folder, not tracked by git
```

## Important notice on comparison methods

The comparison methods are independent works by their respective authors. To respect the original authors' rights and licenses, this public repository **does not redistribute the official implementations** of third-party baselines, including Kalman smoother, SAITS, GRIN, SPIN, CSDI, BayOTIDE, Mamba, LinOSS, TIMBA, KEDGN, HyperIMTS, and ImDiffusion.

Each baseline has an independent `.py` adapter under `baselines/`. These adapter files are intentionally lightweight placeholders. Please read the original paper of each method, obtain the official implementation from the original authors' official repository or release channel, and then complete the corresponding local adapter. Do not commit third-party source code into this repository unless its license explicitly permits redistribution and the original license notice is preserved.

See [`docs/baseline_notice.md`](docs/baseline_notice.md) for the recommended reproduction workflow.

## Installation

```bash
git clone <your-public-repo-url>.git
cd PILOT-Telemetry-Recovery
conda create -n pilot python=3.10 -y
conda activate pilot
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build according to your GPU and CUDA version when GPU training is needed.

## Data format

The default input is an Excel file:

- rows are time steps;
- the first column is time or timestamp;
- the remaining columns are telemetry channels.

Example:

| time | sensor_1 | sensor_2 | ... |
|---|---:|---:|---:|
| 0 | 0.12 | 1.04 | ... |
| 1 | 0.15 | 1.02 | ... |

If your file has a named time column, set `time_col` in the config. If your file has no time column, set `first_col_is_time: false`.

## Quick demo

```bash
python examples/run_demo.py
```

This creates a synthetic Excel file and runs a two-epoch CPU smoke experiment. Results are written to `outputs/demo/results/`.

## Train and test on one Excel file

```bash
python scripts/run_train_test.py \
  --config configs/default.yaml \
  --excel_path /path/to/your_dataset.xlsx \
  --output_dir outputs/my_dataset \
  --models PILOT-Full
```

To run full ablation:

```bash
python scripts/run_train_test.py \
  --excel_path /path/to/your_dataset.xlsx \
  --output_dir outputs/ablation \
  --models PILOT-Temporal,PILOT-w/o-Graph,PILOT-w/o-Separator,PILOT-Full
```

## Run a folder of Excel datasets

```bash
python scripts/run_train_test.py \
  --dataset_folder /path/to/excel_folder \
  --output_dir outputs/folder_run \
  --models PILOT-Full
```

Each dataset gets an independent result subfolder.

## Output files

For each dataset, the pipeline exports:

- `metrics_summary.csv` and `metrics_summary.xlsx`;
- `per_channel_rmse.csv` and `per_channel_rmse.xlsx`;
- `trained_models/<model>.pt`;
- `training_history/<model>_history.csv`;
- `dataset_details.json`;
- `adjacency_global.csv`, `adjacency_local.csv`, `adjacency_cross.csv`;
- `output_manifest.json`;
- optional prediction CSV/XLSX files.

## Model components

The main implementation is in `pilot_recovery/models/pilot.py`:

1. `PILOTMixedSpectrumCell`: event-time mixed-spectrum state flow driven by channel staleness and availability.
2. `PILOTReliabilityHypergraph`: reliability-conditioned local/cross/global hypergraph coupling.
3. `PILOTProjectedSeparator`: finite-window projected innovation separation for anomaly-retentive recovery.
4. `PILOTObserver`: full observer producing reconstruction, anomaly evidence, and uncertainty.

## Citation

This paper is currently under review. The citation metadata for the final publication will be updated after acceptance.

## License

The PILOT implementation in this repository is released under the MIT License. Third-party baselines are not redistributed and remain governed by their own authors' licenses.

# Migration from the previous single-file script

The previous script placed data preparation, baselines, HOSG/PILOT model definitions, training, testing, and exporting in one `.py` file. This public GitHub version is reorganized as follows:

| Previous responsibility | New location |
|---|---|
| `Config` | `pilot_recovery/config.py` |
| Excel reading and missingness construction | `pilot_recovery/data.py` |
| HOSG-style proposed model registration | Replaced by `PILOTObserver` in `pilot_recovery/models/pilot.py` |
| Model registry | `pilot_recovery/models/registry.py` |
| Loss function | `pilot_recovery/losses.py` |
| Training loop | `pilot_recovery/trainer.py` |
| Testing and metrics | `pilot_recovery/metrics.py` |
| Result export | `pilot_recovery/exports.py` |
| `main()` | `scripts/run_train_test.py` |
| Third-party comparison methods | `baselines/<method>.py` adapter placeholders |

The public model registry now exposes only `PILOT-Full` and its ablation variants by default. Third-party comparison methods are not redistributed; use their adapter files after obtaining official code from the original authors.

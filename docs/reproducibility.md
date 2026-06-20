# Reproducibility checklist

- Use identical train/validation/test chronological splits for all methods.
- Use the same artificial missingness masks for all methods.
- Report whether a baseline is official code, reimplementation, or local adapter.
- Save `output_manifest.json`, model checkpoints, and `metrics_summary.csv` for each dataset.
- Report mean and standard deviation across seeds when possible.
- Do not tune baseline hyperparameters on the test set.

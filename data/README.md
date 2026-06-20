# Data format

Input files are Excel workbooks (`.xlsx` or `.xls`). The default format is:

- rows: time steps;
- column 1: time or timestamp;
- columns 2..end: telemetry sensors/channels.

If your file has a named time column, set `time_col` in the YAML config. If your file has no time column, set `first_col_is_time: false`.

Raw telemetry datasets are not included in this public repository. Put local datasets in this folder or pass `--excel_path` / `--dataset_folder` at runtime.

# -*- coding: utf-8 -*-
"""Export dataset split/meta information without training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot_recovery.config import load_config
from pilot_recovery.data import prepare_dataset
from pilot_recovery.exports import save_dataset_details
from pilot_recovery.utils import ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--excel_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="outputs/splits")
args = parser.parse_args()

cfg = load_config(args.config, overrides={"excel_path": args.excel_path, "output_dir": args.output_dir})
data = prepare_dataset(cfg)
save_dataset_details(data, cfg, ensure_dir(args.output_dir))
print(f"Saved split details to {args.output_dir}")

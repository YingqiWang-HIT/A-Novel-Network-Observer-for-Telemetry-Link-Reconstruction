# -*- coding: utf-8 -*-
"""Convenience wrapper for folder experiments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name("run_train_test.py")
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))

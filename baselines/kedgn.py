# -*- coding: utf-8 -*-
"""Adapter placeholder for KEDGN.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class KEDGNAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="KEDGN",
        paper="KEDGN graph-based telemetry/time-series imputation method; consult the original publication",
    )


def build_model(*args, **kwargs):
    return KEDGNAdapter(*args, **kwargs)

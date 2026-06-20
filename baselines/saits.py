# -*- coding: utf-8 -*-
"""Adapter placeholder for SAITS.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class SAITSAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="SAITS",
        paper="SAITS: Self-Attention-based Imputation for Time Series",
    )


def build_model(*args, **kwargs):
    return SAITSAdapter(*args, **kwargs)

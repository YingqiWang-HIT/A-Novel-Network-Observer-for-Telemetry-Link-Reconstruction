# -*- coding: utf-8 -*-
"""Adapter placeholder for SPIN.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class SPINAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="SPIN",
        paper="SPIN: Spatial-Temporal Graph Imputation Network",
    )


def build_model(*args, **kwargs):
    return SPINAdapter(*args, **kwargs)

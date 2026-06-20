# -*- coding: utf-8 -*-
"""Adapter placeholder for HyperIMTS.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class HyperIMTSAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="HyperIMTS",
        paper="HyperIMTS hypergraph irregular multivariate time-series imputation method",
    )


def build_model(*args, **kwargs):
    return HyperIMTSAdapter(*args, **kwargs)

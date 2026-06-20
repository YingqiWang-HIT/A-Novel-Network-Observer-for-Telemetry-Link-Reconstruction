# -*- coding: utf-8 -*-
"""Adapter placeholder for BayOTIDE.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class BayOTIDEAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="BayOTIDE",
        paper="BayOTIDE: Bayesian Online Time-series Imputation with Deep Dynamics",
    )


def build_model(*args, **kwargs):
    return BayOTIDEAdapter(*args, **kwargs)

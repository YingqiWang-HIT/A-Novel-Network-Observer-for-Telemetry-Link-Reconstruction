# -*- coding: utf-8 -*-
"""Adapter placeholder for GRIN.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class GRINAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="GRIN",
        paper="GRIN: Graph Recurrent Imputation Network",
    )


def build_model(*args, **kwargs):
    return GRINAdapter(*args, **kwargs)

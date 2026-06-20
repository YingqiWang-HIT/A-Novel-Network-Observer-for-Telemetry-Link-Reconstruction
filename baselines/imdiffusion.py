# -*- coding: utf-8 -*-
"""Adapter placeholder for ImDiffusion.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class ImDiffusionAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="ImDiffusion",
        paper="ImDiffusion anomaly-aware/diffusion-based imputation method",
    )


def build_model(*args, **kwargs):
    return ImDiffusionAdapter(*args, **kwargs)

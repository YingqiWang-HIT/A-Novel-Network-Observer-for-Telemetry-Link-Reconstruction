# -*- coding: utf-8 -*-
"""Adapter placeholder for CSDI.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class CSDIAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="CSDI",
        paper="CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
    )


def build_model(*args, **kwargs):
    return CSDIAdapter(*args, **kwargs)

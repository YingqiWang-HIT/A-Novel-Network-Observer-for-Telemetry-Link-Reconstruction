# -*- coding: utf-8 -*-
"""Adapter placeholder for Kalman smoother.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class KalmanSmootherAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="Kalman smoother",
        paper="Classical Kalman filtering and smoothing for state-space models",
    )


def build_model(*args, **kwargs):
    return KalmanSmootherAdapter(*args, **kwargs)

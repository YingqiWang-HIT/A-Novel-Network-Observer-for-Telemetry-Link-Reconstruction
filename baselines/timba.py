# -*- coding: utf-8 -*-
"""Adapter placeholder for TIMBA.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class TIMBAAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="TIMBA",
        paper="TIMBA time-series model; consult the original publication for the exact implementation",
    )


def build_model(*args, **kwargs):
    return TIMBAAdapter(*args, **kwargs)

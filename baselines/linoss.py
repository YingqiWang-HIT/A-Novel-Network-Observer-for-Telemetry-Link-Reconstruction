# -*- coding: utf-8 -*-
"""Adapter placeholder for LinOSS.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class LinOSSAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="LinOSS",
        paper="Linear Oscillatory State-Space model for long-range time-series modeling",
    )


def build_model(*args, **kwargs):
    return LinOSSAdapter(*args, **kwargs)

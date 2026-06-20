# -*- coding: utf-8 -*-
"""Adapter placeholder for Mamba.

This file intentionally does not contain the authors' official implementation.
See README.md and docs/baseline_notice.md before using this baseline.
"""
from __future__ import annotations

from .adapter_base import BaselineSource, ExternalBaselineAdapter


class MambaAdapter(ExternalBaselineAdapter):
    SOURCE = BaselineSource(
        name="Mamba",
        paper="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    )


def build_model(*args, **kwargs):
    return MambaAdapter(*args, **kwargs)

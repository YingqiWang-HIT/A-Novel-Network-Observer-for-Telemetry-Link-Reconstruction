# -*- coding: utf-8 -*-
"""Adapter interface for third-party comparison methods.

The public PILOT repository does not redistribute official code of comparison methods.
Users should obtain each baseline from its original paper/official repository and implement
this small adapter interface locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class BaselineUnavailableError(RuntimeError):
    pass


@dataclass
class BaselineSource:
    name: str
    paper: str
    note: str = "Official implementation is not redistributed in this repository. Please consult the original paper and the authors' official release."
    official_url: Optional[str] = None


class ExternalBaselineAdapter:
    SOURCE = BaselineSource(name="Unknown", paper="Unknown")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def source(cls) -> BaselineSource:
        return cls.SOURCE

    def fit(self, *args, **kwargs):
        raise BaselineUnavailableError(self._message())

    def predict(self, *args, **kwargs):
        raise BaselineUnavailableError(self._message())

    def _message(self) -> str:
        s = self.source()
        return (
            f"{s.name} is a third-party comparison method. Its official implementation is not bundled "
            f"because the original authors retain rights over their code. Please read the original paper: "
            f"{s.paper}. Then place the official implementation under baselines/external/{s.name}/ "
            f"and complete this adapter."
        )

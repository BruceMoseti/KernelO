"""Configuration selection at call time.

This is the piece that makes the cache worth having. A model does not want to
tune a GEMM on every forward pass, and it does not want the untuned default
either. It wants to ask "has this shape been tuned on this GPU?" and get an
answer in microseconds.

    hit  -> the configuration tuning chose
    miss -> the operator's default, with the miss reported so it is not silent

Nothing here tunes. Selecting a configuration must never trigger a
thirty-second compile sweep inside someone's inference loop; tuning is an
explicit step (``kernelforge tune``) whose output this reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernelforge.kernels.base import Operator
from kernelforge.runtime.env import device_caps
from kernelforge.tuning.cache import ConfigCache
from kernelforge.tuning.config import KernelConfig, Problem

SOURCE_CACHE = "cache"
SOURCE_DEFAULT = "default"


@dataclass(frozen=True)
class Selection:
    config: KernelConfig
    source: str
    device_key: str

    @property
    def tuned(self) -> bool:
        return self.source == SOURCE_CACHE

    def describe(self) -> str:
        if self.tuned:
            return f"tuned configuration for {self.device_key}"
        return f"default configuration (no cache entry for {self.device_key})"


def select_config(
    operator: Operator,
    problem: Problem,
    *,
    cache: ConfigCache | None = None,
    device: torch.device | str | None = None,
) -> Selection:
    """Pick the configuration to run ``problem`` with."""
    device_key = device_caps(device).key
    resolved = cache if cache is not None else ConfigCache()
    cached = resolved.get(problem, device_key)
    if cached is not None:
        return Selection(config=cached, source=SOURCE_CACHE, device_key=device_key)
    return Selection(
        config=operator.default_config(problem),
        source=SOURCE_DEFAULT,
        device_key=device_key,
    )

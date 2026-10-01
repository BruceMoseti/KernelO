"""The contract a tunable operator fulfils.

The tuner knows nothing about GEMMs. It asks an operator for a search space,
for inputs, for a reference result and for a way to run one configuration;
everything else -- filtering, verification, timing, ranking, persistence -- is
generic. Adding an operator therefore means implementing this interface, not
touching the tuner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import SearchSpace


class Operator(ABC):
    """One tunable operation."""

    name: str
    #: Preferred key order when rendering a config in the CLI.
    config_order: tuple[str, ...] = ()

    @abstractmethod
    def search_space(self) -> SearchSpace: ...

    @abstractmethod
    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        """Build reproducible inputs for ``problem`` on ``device``."""

    @abstractmethod
    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The PyTorch implementation this kernel must agree with."""

    @abstractmethod
    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute the Triton kernel with ``config``."""

    @abstractmethod
    def default_config(self, problem: Problem) -> KernelConfig:
        """A fixed, always-feasible configuration.

        This is the untuned Triton baseline that tuning is measured against,
        and the fallback when the cache misses and tuning is not requested.
        """

    @abstractmethod
    def flops(self, problem: Problem) -> int: ...

    @abstractmethod
    def bytes_moved(self, problem: Problem) -> int:
        """Minimum global-memory traffic, in bytes, for a perfect kernel.

        Counts each input read once and each output written once. Real traffic
        is higher; the gap is what the memory analysis in the case studies is
        about.
        """

    def baselines(
        self, problem: Problem, inputs: tuple[torch.Tensor, ...]
    ) -> dict[str, Callable[[], torch.Tensor]]:
        """Implementations to compare against, keyed by label.

        ``torch_eager`` is always present. Operators add ``torch_compile`` and
        ``triton_autotune`` where a meaningful comparison exists.
        """
        return {"torch_eager": lambda: self.reference(*inputs)}

    def is_memory_bound(self) -> bool:
        """Whether GB/s or TFLOP/s is the headline metric for this operator."""
        return False


def compiled_baseline(
    fn: Callable[..., torch.Tensor], inputs: tuple[torch.Tensor, ...], *, mode: str | None = None
) -> Callable[[], torch.Tensor]:
    """Wrap ``fn`` in ``torch.compile``.

    Nothing is compiled until the first call. Callers verify a baseline's
    output before timing it, so Dynamo tracing and Inductor codegen still land
    outside the measurement, and a compile failure surfaces at that one
    baseline instead of while every baseline is being built.
    """
    compiled = torch.compile(fn, mode=mode) if mode else torch.compile(fn)

    def run() -> torch.Tensor:
        return compiled(*inputs)

    return run

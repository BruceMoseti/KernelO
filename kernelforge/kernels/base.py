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

    def exact_reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        """:meth:`reference` computed in float64 from the same inputs.

        Upcasting is lossless and float64's own rounding is negligible, so this
        is exact for the inputs the kernel received, and
        :func:`~kernelforge.testing.verify` holds a kernel to an elementwise
        bound against it.
        """
        return self.reference(*(t.to(torch.float64) for t in inputs))

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
    """Wrap ``fn`` in ``torch.compile`` and warm it up.

    Compilation is triggered eagerly here so that the first measured iteration
    is not dominated by Dynamo tracing and Inductor codegen.
    """
    compiled = torch.compile(fn, mode=mode) if mode else torch.compile(fn)

    def run() -> torch.Tensor:
        return compiled(*inputs)

    run()
    return run

"""A transformer block with swappable kernels.

The last thing worth measuring, and the one most likely to be humbling: a
kernel that is twice as fast does not make the model twice as fast. A pre-norm
decoder block spends most of its time in the attention and projection GEMMs,
so replacing the two normalisations and one activation changes end-to-end
latency by whatever share of the block those operators held -- which is the
point. Amdahl's law is easy to recite and easy to forget while staring at a
microbenchmark.

What is swapped, and what is not:

* **Swapped.** Both RMSNorms, and the MLP up-projection, whose bias and GELU
  fold into the GEMM epilogue. These are the operators KernelForge optimises.
* **Not swapped.** Attention itself stays on
  ``scaled_dot_product_attention``. Writing a competitive fused attention
  kernel is a project of its own and explicitly out of scope; borrowing
  PyTorch's and being honest about it is better than a slow one here.
* **Not swapped.** The QKV, output and down projections stay on
  ``nn.Linear``, so the comparison isolates the fused epilogue rather than
  measuring this GEMM against cuBLAS four more times.

Both backends share one set of weights -- ``backend`` is a mutable attribute
on a single module, not two modules -- so a parity check cannot be fooled by
differing initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kernelforge.kernels.fused_linear import fused_linear_gelu
from kernelforge.kernels.rmsnorm import DEFAULT_EPS, rmsnorm, rmsnorm_reference
from kernelforge.runtime.dispatch import select_config
from kernelforge.tuning.cache import ConfigCache
from kernelforge.tuning.config import KernelConfig, Problem

BACKEND_TORCH = "torch"
BACKEND_KERNELFORGE = "kernelforge"
BACKENDS = (BACKEND_TORCH, BACKEND_KERNELFORGE)


@dataclass(frozen=True)
class BlockConfig:
    """Geometry of one decoder block.

    The default is Llama-2-7B's: 4096 hidden, 32 heads, 11008 intermediate.
    The MLP here is GELU rather than SwiGLU, because the fused kernel is a
    bias+GELU epilogue, but the matrix shapes -- which is what the performance
    question turns on -- are the real ones.
    """

    hidden: int = 4096
    heads: int = 32
    intermediate: int = 11008
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise ValueError(f"hidden {self.hidden} is not divisible by {self.heads} heads")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


class TransformerBlock(nn.Module):
    """Pre-norm decoder block whose normalisation and MLP activation are swappable."""

    def __init__(
        self,
        config: BlockConfig | None = None,
        *,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        backend: str = BACKEND_TORCH,
        cache: ConfigCache | None = None,
    ) -> None:
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.config = config or BlockConfig()
        self.backend = backend
        self._cache = cache
        # Configuration lookups touch the filesystem, so they are resolved once
        # per shape rather than on every forward pass.
        self._selected: dict[tuple[str, int, int], KernelConfig] = {}

        hidden, intermediate = self.config.hidden, self.config.intermediate
        factory = {"device": device, "dtype": dtype}
        self.attn_norm_weight = nn.Parameter(torch.ones(hidden, **factory))
        self.mlp_norm_weight = nn.Parameter(torch.ones(hidden, **factory))
        self.qkv_proj = nn.Linear(hidden, 3 * hidden, bias=False, **factory)
        self.out_proj = nn.Linear(hidden, hidden, bias=False, **factory)
        self.mlp_up = nn.Linear(hidden, intermediate, bias=True, **factory)
        self.mlp_down = nn.Linear(intermediate, hidden, bias=False, **factory)

    # --- kernel selection ----------------------------------------------
    def _config_for(self, operation: str, rows: int, cols: int, dtype: torch.dtype) -> KernelConfig:
        key = (operation, rows, cols)
        if key not in self._selected:
            if operation == "rmsnorm":
                problem = Problem.create(operation, dtype, rows=rows, cols=cols)
            else:
                problem = Problem.create(
                    operation, dtype, M=rows, N=self.config.intermediate, K=cols
                )
            from kernelforge.kernels import get_operator

            self._selected[key] = select_config(
                get_operator(operation), problem, cache=self._cache
            ).config
        return self._selected[key]

    # --- swappable pieces ----------------------------------------------
    def _norm(self, flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.backend == BACKEND_TORCH:
            return rmsnorm_reference(flat, weight, eps=self.config.eps)
        config = self._config_for("rmsnorm", flat.shape[0], flat.shape[1], flat.dtype)
        return rmsnorm(flat, weight, eps=self.config.eps, config=config)

    def _mlp_activation(self, flat: torch.Tensor) -> torch.Tensor:
        if self.backend == BACKEND_TORCH:
            return F.gelu(self.mlp_up(flat), approximate="tanh")
        config = self._config_for("fused_linear", flat.shape[0], flat.shape[1], flat.dtype)
        # nn.Linear stores (out, in); the kernel takes (K, N) and reads strides,
        # so the transposed view needs no copy.
        return fused_linear_gelu(flat, self.mlp_up.weight.t(), self.mlp_up.bias, config=config)

    # --- forward -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, seq, hidden), got {tuple(x.shape)}")
        batch, seq, hidden = x.shape
        if hidden != self.config.hidden:
            raise ValueError(f"expected hidden {self.config.hidden}, got {hidden}")
        heads, head_dim = self.config.heads, self.config.head_dim

        normed = self._norm(x.reshape(batch * seq, hidden), self.attn_norm_weight)
        qkv = self.qkv_proj(normed.view(batch, seq, hidden))
        q, k, v = (
            t.view(batch, seq, heads, head_dim).transpose(1, 2) for t in qkv.split(hidden, dim=-1)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq, hidden)
        x = x + self.out_proj(attended)

        normed = self._norm(x.reshape(batch * seq, hidden), self.mlp_norm_weight)
        hidden_states = self._mlp_activation(normed)
        return x + self.mlp_down(hidden_states).view(batch, seq, hidden)


def benchmark_block(
    config: BlockConfig | None = None,
    *,
    batch: int = 1,
    seq: int = 2048,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    warmup: int = 25,
    iterations: int = 100,
    cache: ConfigCache | None = None,
) -> dict[str, object]:
    """Time one block with each backend, on identical weights and inputs."""
    from kernelforge.benchmark.runner import benchmark

    block = TransformerBlock(config, dtype=dtype, device=device, cache=cache)
    x = torch.randn(batch, seq, block.config.hidden, device=device, dtype=dtype)

    results: dict[str, object] = {}
    with torch.inference_mode():
        for backend in BACKENDS:
            block.backend = backend
            results[backend] = benchmark(
                lambda: block(x),
                warmup=warmup,
                iterations=iterations,
                device=device,
                # The block's working set is far larger than L2, so flushing
                # would only add noise to a measurement that is already cold.
                flush_l2=False,
            )
    return results

"""Fused linear + bias + GELU: ``y = gelu(x @ w + bias)``.

The unfused sequence materialises two full ``M x N`` intermediates:

    x @ w  -> tmp   (write M*N)
    tmp + bias -> tmp2   (read M*N, write M*N)
    gelu(tmp2) -> y  (read M*N, write M*N)

Three kernel launches and 5*M*N elements of traffic through the output. The
epilogue arithmetic is trivial -- a few FLOPs per element -- so every one of
those extra passes is pure bandwidth cost. Folding the bias and the activation
into the GEMM's epilogue, while the tile is still in registers, leaves one
launch and M*N of output traffic. For M=4096, N=11008 in fp16 that is 344 MiB
of avoided round trips.

**GELU form.** This is the tanh approximation, which is what transformer
implementations that specify an approximation use:

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Rewritten through ``tanh(z) = 2*sigmoid(2z) - 1`` it collapses to

    gelu(x) = x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))

which needs one ``exp`` instead of a ``tanh``, depends on no Triton version's
libdevice bindings, and cannot overflow: a large negative argument sends the
exponential to ``+inf`` and the result to zero, which is the correct limit.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from kernelforge.benchmark import metrics
from kernelforge.kernels.base import Operator, compiled_baseline
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import FusedLinearSearchSpace

# 2 * sqrt(2/pi), with the factor of two from the sigmoid identity folded in.
# Wrapped in tl.constexpr because a @triton.jit kernel cannot read an ordinary
# Python global.
_GELU_COEFF = tl.constexpr(1.5957691216057308)


@triton.jit
def fused_linear_gelu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_wn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # --- fused epilogue, on the tile still resident in registers ---------
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_yn, mask=offs_yn < N, other=0.0).to(tl.float32)
    acc = accumulator + bias[None, :]
    inner = _GELU_COEFF * (acc + 0.044715 * acc * acc * acc)
    acc = acc / (1.0 + tl.exp(-inner))

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)


DEFAULT_CONFIG = KernelConfig(
    "fused_linear", BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3
)


def fused_linear_gelu(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    *,
    config: KernelConfig | None = None,
) -> torch.Tensor:
    """``gelu(x @ w + bias)`` in one kernel.

    ``w`` is ``(K, N)``, matching ``torch.matmul``. An ``nn.Linear`` holds its
    weight as ``(out_features, in_features)``, so callers pass ``weight.t()``;
    the kernel takes strides rather than assuming contiguity, so the
    transposed view needs no copy.
    """
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"expected 2D x and w, got {tuple(x.shape)} and {tuple(w.shape)}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} @ {tuple(w.shape)}")
    if bias.shape != (w.shape[1],):
        raise ValueError(f"bias must have shape ({w.shape[1]},), got {tuple(bias.shape)}")
    if not (x.dtype == w.dtype == bias.dtype):
        raise ValueError(f"dtype mismatch: {x.dtype}, {w.dtype}, {bias.dtype}")

    cfg = config or DEFAULT_CONFIG
    m, k = x.shape
    _, n = w.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    fused_linear_gelu_kernel[grid](
        x,
        w,
        bias,
        y,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        **cfg.meta,
        **cfg.launch,
    )
    return y


def linear_gelu_reference(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The unfused sequence, exactly as it would be written in a model."""
    return torch.nn.functional.gelu(torch.matmul(x, w) + bias, approximate="tanh")


class FusedLinearOperator(Operator):
    name = "fused_linear"
    config_order = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages")

    def search_space(self) -> FusedLinearSearchSpace:
        return FusedLinearSearchSpace()

    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device).manual_seed(seed)
        dims = problem.dims_dict
        x = torch.randn(dims["M"], dims["K"], device=device, dtype=problem.dtype, generator=gen)
        # Scaled like an initialised weight so that the pre-activation lands in
        # the range where GELU is actually nonlinear; with unit-variance
        # weights and K=4096 the activation would saturate to identity and the
        # test would not exercise the curve.
        w = torch.randn(dims["K"], dims["N"], device=device, dtype=torch.float32, generator=gen)
        w = (w * (dims["K"] ** -0.5)).to(problem.dtype)
        bias = torch.randn(dims["N"], device=device, dtype=problem.dtype, generator=gen)
        return x, w, bias

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, w, bias = inputs
        return linear_gelu_reference(x, w, bias)

    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        x, w, bias = inputs
        return fused_linear_gelu(x, w, bias, config=config)

    def default_config(self, problem: Problem) -> KernelConfig:
        return DEFAULT_CONFIG

    def flops(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.matmul_flops(dims["M"], dims["N"], dims["K"])

    def bytes_moved(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.fused_linear_bytes(
            dims["M"], dims["N"], dims["K"], problem.itemsize, fused=True
        )

    def baselines(
        self, problem: Problem, inputs: tuple[torch.Tensor, ...]
    ) -> dict[str, Callable[[], torch.Tensor]]:
        x, w, bias = inputs
        return {
            "torch_eager": lambda: linear_gelu_reference(x, w, bias),
            "torch_compile": compiled_baseline(linear_gelu_reference, inputs),
            "triton_baseline": lambda: fused_linear_gelu(x, w, bias, config=DEFAULT_CONFIG),
        }

"""Derived performance metrics.

"35% faster" says nothing about whether a kernel is near the hardware limit.
These functions convert a latency into the two numbers that do: TFLOP/s against
the card's peak compute for the GEMMs, and GB/s against its peak bandwidth for
the reductions.

Byte counts are *minimum* traffic -- each input read once, each output written
once -- so measured bandwidth can legitimately come out below the achievable
peak when a kernel re-reads data. That gap is a result, not an error in the
model, and it is what the memory analysis in the case studies examines.
"""

from __future__ import annotations

from dataclasses import dataclass


def matmul_flops(m: int, n: int, k: int) -> int:
    """``2*M*N*K``: one multiply and one add per inner-product term."""
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int, itemsize: int) -> int:
    """Minimum traffic for a GEMM: read A and B once, write C once.

    A tiled GEMM necessarily re-reads the operands (each A tile is used by
    every column block), so the true figure is higher by roughly a factor of
    ``N/BLOCK_N`` for A and ``M/BLOCK_M`` for B. This bound is what
    arithmetic intensity is measured against.
    """
    return (m * k + k * n + m * n) * itemsize


def elementwise_bytes(elements: int, itemsize: int, *, tensors: int) -> int:
    return elements * itemsize * tensors


def vector_add_bytes(n: int, itemsize: int) -> int:
    """Two reads and one write per element."""
    return 3 * n * itemsize


def rmsnorm_bytes(rows: int, cols: int, itemsize: int) -> int:
    """Read x, write y, read gamma once per launch."""
    return (2 * rows * cols + cols) * itemsize


def fused_linear_bytes(m: int, n: int, k: int, itemsize: int, *, fused: bool) -> int:
    """Traffic for ``gelu(x @ w + bias)``.

    Unfused, the bias add and the activation each read and rewrite the full
    ``M x N`` intermediate, for ``5*M*N`` of output traffic against ``M*N``
    fused. The difference is the quantity the fusion case study predicts and
    then checks against the profiler.
    """
    operands = m * k + k * n + n
    output = m * n if fused else 5 * m * n
    return (operands + output) * itemsize


def tflops(flops: int, ms: float) -> float:
    if ms <= 0:
        raise ValueError(f"latency must be positive, got {ms}")
    return flops / (ms * 1e-3) / 1e12


def gbps(num_bytes: int, ms: float) -> float:
    if ms <= 0:
        raise ValueError(f"latency must be positive, got {ms}")
    return num_bytes / (ms * 1e-3) / 1e9


def arithmetic_intensity(flops: int, num_bytes: int) -> float:
    """FLOP per byte.

    Compared against a GPU's ratio of peak FLOP/s to peak GB/s, this is what
    decides whether a kernel can possibly be compute bound. An A100 at
    312 TFLOP/s fp16 and 2039 GB/s has a ridge point near 153 FLOP/byte: below
    that, no amount of kernel tuning gets past the memory system.
    """
    if num_bytes <= 0:
        raise ValueError(f"byte count must be positive, got {num_bytes}")
    return flops / num_bytes


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        raise ValueError(f"latency must be positive, got {candidate_ms}")
    return baseline_ms / candidate_ms


@dataclass(frozen=True)
class PublishedPeak:
    """NVIDIA's published dense peak rates for one GPU, at boost clock.

    Spec-sheet figures, not measurements. A measured TFLOP/s or GB/s divided by
    one of them is the hardware utilisation, and no kernel reaches 100%.
    """

    #: FP16 and BF16 Tensor Core math with FP32 accumulate, which is what
    #: ``tl.dot`` does here. GeForce cards run it at half their FP16-accumulate
    #: rate, so that figure would overstate the roof.
    tensor_tflops: float
    #: Without Tensor Cores: the kernels pin fp32 ``tl.dot`` to IEEE, not TF32.
    fp32_tflops: float
    dram_gbps: float
    source: str

    def rate(self, dtype: str, *, memory_bound: bool) -> float:
        """The roof that a kernel's GB/s, or TFLOP/s, is a fraction of."""
        if memory_bound:
            return self.dram_gbps
        return self.fp32_tflops if dtype == "fp32" else self.tensor_tflops


_A100_DATASHEET = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/"
    "nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
)
_ADA_WHITEPAPER = (
    "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
)

#: Keyed by the device name CUDA reports. A GPU missing here gets no
#: utilisation figure rather than an estimated one.
PUBLISHED_PEAKS: dict[str, PublishedPeak] = {
    "NVIDIA A100-PCIE-40GB": PublishedPeak(312.0, 19.5, 1555.0, _A100_DATASHEET),
    "NVIDIA A100 80GB PCIe": PublishedPeak(312.0, 19.5, 1935.0, _A100_DATASHEET),
    "NVIDIA A100-SXM4-40GB": PublishedPeak(312.0, 19.5, 1555.0, _A100_DATASHEET),
    "NVIDIA A100-SXM4-80GB": PublishedPeak(312.0, 19.5, 2039.0, _A100_DATASHEET),
    "NVIDIA GeForce RTX 4090": PublishedPeak(165.2, 82.6, 1008.0, _ADA_WHITEPAPER),
}

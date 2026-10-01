"""MatMul correctness.

Covers the spec's torture shapes, seeded random shapes, and every tile shape in the tuner's
search space. Operands are also placed inside NaN-filled guard buffers, so an out-of-bounds load
turns the result into NaN. The output sits inside a buffer filled with a finite sentinel, so an
out-of-bounds store changes a guard cell. Both are detected.
"""

import pytest
import torch
import triton

from kernelforge.kernels.matmul import matmul
from kernelforge.testing import verify

# Triton 3.8's interpreter multiplies bf16 operands as raw uint16 bit patterns, so bf16 needs a GPU.
DTYPES = [torch.float32, torch.float16, pytest.param(torch.bfloat16, marks=pytest.mark.gpu)]
CPU_DTYPES = [torch.float32, torch.float16]
ALL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_generator = torch.Generator().manual_seed(0)
RANDOM_SHAPES = [
    tuple(torch.randint(1, 160, (3,), generator=_generator).tolist()) for _ in range(40)
]
SEARCH_SPACE_TILES = [
    (block_m, block_n, block_k)
    for block_m in (16, 32, 64, 128)
    for block_n in (16, 32, 64, 128)
    for block_k in (16, 32, 64)
]


def _operands(
    m: int, n: int, k: int, dtype: torch.dtype, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=generator).to(device=device, dtype=dtype)
    b = torch.randn(k, n, generator=generator).to(device=device, dtype=dtype)
    return a, b


def _check(a: torch.Tensor, b: torch.Tensor, **config: int) -> None:
    try:
        out = matmul(a, b, **config)
    except triton.runtime.errors.OutOfResources as error:
        pytest.skip(f"config does not fit this GPU: {error}")
    result = verify(a.double() @ b.double(), out, a.dtype)
    assert result.passed, result.describe()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(128, 128, 128), (511, 769, 1025)])
def test_spec_torture_shapes(
    shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    _check(*_operands(*shape, dtype, device))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (2048, 768, 4096), (4096, 4096, 4096)])
def test_spec_torture_shapes_large(shape: tuple[int, int, int], dtype: torch.dtype) -> None:
    _check(*_operands(*shape, dtype, torch.device("cuda")))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_random_shapes(
    shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    _check(*_operands(*shape, dtype, device, seed=sum(shape)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tile", SEARCH_SPACE_TILES)
def test_every_search_space_tile_shape(
    tile: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    # No dimension is a multiple of any block size, so every tile shape exercises all masks.
    block_m, block_n, block_k = tile
    a, b = _operands(67, 45, 83, dtype, device)
    _check(a, b, block_m=block_m, block_n=block_n, block_k=block_k)


def test_partial_final_group_band(device: torch.device) -> None:
    # 19 tile rows with BLOCK_M=16 form program bands of 8, 8 and 3 rows.
    a, b = _operands(299, 40, 50, torch.float32, device)
    _check(a, b, block_m=16, block_n=16, block_k=16)


# At least the largest block size, so a whole-tile overrun lands in guard cells rather than
# beyond the buffer.
GUARD = 128
# Exactly representable in every dtype, and not a value a stray store would produce (stray
# stores computed from NaN-guarded operands write NaN).
OUT_SENTINEL = -1024.0


def _guarded(
    rows: int, cols: int, dtype: torch.dtype, device: torch.device, fill: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """A rows x cols view surrounded on all sides by GUARD rows and columns of `fill`."""
    shape = (rows + 2 * GUARD, cols + 2 * GUARD)
    buffer = torch.full(shape, fill, dtype=dtype, device=device)
    return buffer, buffer[GUARD : GUARD + rows, GUARD : GUARD + cols]


def _guard_cells_intact(buffer: torch.Tensor, view: torch.Tensor) -> bool:
    outside = torch.ones_like(buffer, dtype=torch.bool)
    outside[GUARD : GUARD + view.shape[0], GUARD : GUARD + view.shape[1]] = False
    return bool((buffer[outside] == OUT_SENTINEL).all())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b_layout", ["row-major", "column-major"])
@pytest.mark.parametrize("tile", [(16, 16, 16), (32, 32, 32), (64, 32, 16)])
@pytest.mark.parametrize("shape", [(33, 47, 61), (1, 1, 1), (70, 17, 129)])
def test_no_out_of_bounds_loads_or_stores(
    shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    b_layout: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    m, n, k = shape
    a_values, b_values = _operands(m, n, k, dtype, device)
    nan = float("nan")
    _, a = _guarded(m, k, dtype, device, fill=nan)
    a.copy_(a_values)
    if b_layout == "row-major":
        _, b = _guarded(k, n, dtype, device, fill=nan)
    else:
        _, b_transposed = _guarded(n, k, dtype, device, fill=nan)
        b = b_transposed.t()
    b.copy_(b_values)
    out_buffer, out = _guarded(m, n, dtype, device, fill=OUT_SENTINEL)

    block_m, block_n, block_k = tile
    matmul(a, b, block_m=block_m, block_n=block_n, block_k=block_k, out=out)

    result = verify(a_values.double() @ b_values.double(), out, dtype)
    assert result.passed, result.describe()
    assert _guard_cells_intact(out_buffer, out), "kernel wrote outside the output matrix"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("num_stages", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_warps_and_stages_do_not_change_results(
    num_warps: int, num_stages: int, dtype: torch.dtype
) -> None:
    # The interpreter ignores num_warps and num_stages; on a GPU they change the generated code.
    a, b = _operands(511, 769, 1025, dtype, torch.device("cuda"))
    _check(a, b, block_m=64, block_n=64, block_k=32, num_warps=num_warps, num_stages=num_stages)


def test_rejects_invalid_operands(device: torch.device) -> None:
    a, b = _operands(4, 5, 6, torch.float32, device)
    with pytest.raises(ValueError, match="incompatible shapes"):
        matmul(a, a)
    with pytest.raises(ValueError, match="incompatible shapes"):
        matmul(a[0], b)
    with pytest.raises(ValueError, match="same dtype"):
        matmul(a, b.half())
    with pytest.raises(ValueError, match="unsupported dtype"):
        matmul(a.int(), b.int())
    with pytest.raises(ValueError, match="out must be"):
        matmul(a, b, out=torch.empty(4, 4, device=device))

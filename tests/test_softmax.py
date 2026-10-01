import pytest
import torch

from kernelforge.kernels.softmax import softmax
from kernelforge.testing import verify

# The kernel computes in fp32 after loading, which Triton's interpreter emulates correctly for
# bf16 too, so all three dtypes run on CPU.
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BOUNDARY_SHAPES = [
    (1, 1),
    (1, 2),
    (3, 7),
    (127, 129),
    (4, 257),
    (2, 511),
    (5, 769),
    (3, 1023),
    (2, 1024),
    (2, 2057),
    (2, 4096),
]
_generator = torch.Generator().manual_seed(0)
RANDOM_SHAPES = list(
    zip(
        torch.randint(1, 65, (40,), generator=_generator).tolist(),
        torch.randint(1, 3000, (40,), generator=_generator).tolist(),
        strict=True,
    )
)
SPEC_BENCHMARK_SHAPES = [
    (rows, cols) for rows in (128, 512, 2048, 8192) for cols in (128, 256, 512, 1024, 2048, 4096)
]


def _check(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device, scale: float = 1.0
) -> None:
    generator = torch.Generator().manual_seed(shape[0] * 10_000 + shape[1])
    x = (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)
    result = verify(torch.softmax(x.double(), dim=-1), softmax(x), dtype)
    assert result.passed, result.describe()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", BOUNDARY_SHAPES)
def test_boundary_shapes(shape: tuple[int, int], dtype: torch.dtype, device: torch.device) -> None:
    _check(shape, dtype, device)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_random_shapes(shape: tuple[int, int], device: torch.device) -> None:
    _check(shape, torch.float16, device)


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_inputs_do_not_overflow(dtype: torch.dtype, device: torch.device) -> None:
    # Without subtracting the row max, exp() of these inputs overflows to inf.
    _check((16, 1000), dtype, device, scale=1000.0)


def test_rows_with_padding_between_them(device: torch.device) -> None:
    generator = torch.Generator().manual_seed(0)
    wide = torch.randn(33, 300, generator=generator).to(device)
    x = wide[:, :257]
    result = verify(torch.softmax(x.double(), dim=-1), softmax(x), torch.float32)
    assert result.passed, result.describe()


def test_rejects_unsupported_layouts(device: torch.device) -> None:
    x = torch.randn(4, 8, device=device)
    with pytest.raises(ValueError, match="2-D"):
        softmax(x[0])
    with pytest.raises(ValueError, match="contiguous"):
        softmax(x.t())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SPEC_BENCHMARK_SHAPES)
def test_spec_benchmark_shapes(shape: tuple[int, int], dtype: torch.dtype) -> None:
    _check(shape, dtype, torch.device("cuda"))

import pytest
import torch

from kernelforge.kernels.vector_add import vector_add
from kernelforge.testing import verify

# Triton 3.8's interpreter does bf16 arithmetic on the raw uint16 storage, so bf16 needs a GPU.
DTYPES = [torch.float32, torch.float16, pytest.param(torch.bfloat16, marks=pytest.mark.gpu)]
SPEC_SIZES = [1, 13, 127, 1024, 12345, 1_000_003]
RANDOM_SIZES = sorted(
    set(torch.randint(1, 200_000, (40,), generator=torch.Generator().manual_seed(0)).tolist())
)


def _inputs(
    n: int, dtype: torch.dtype, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=generator).to(device=device, dtype=dtype)
    y = torch.randn(n, generator=generator).to(device=device, dtype=dtype)
    return x, y


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SPEC_SIZES)
def test_spec_sizes(n: int, dtype: torch.dtype, device: torch.device) -> None:
    x, y = _inputs(n, dtype, device)
    out = vector_add(x, y)
    result = verify(x.double() + y.double(), out, dtype)
    assert result.passed, result.describe()
    # Addition is correctly rounded in both implementations, so the bits must match exactly.
    assert torch.equal(out, x + y)


@pytest.mark.parametrize("n", RANDOM_SIZES)
def test_random_sizes(n: int, device: torch.device) -> None:
    x, y = _inputs(n, torch.float16, device, seed=n)
    out = vector_add(x, y)
    result = verify(x.double() + y.double(), out, torch.float16)
    assert result.passed, result.describe()


def test_rejects_mismatched_or_strided_inputs(device: torch.device) -> None:
    x, y = _inputs(16, torch.float32, device)
    with pytest.raises(ValueError, match="same shape"):
        vector_add(x, y[:8])
    with pytest.raises(ValueError, match="same shape"):
        vector_add(x, y.half())
    with pytest.raises(ValueError, match="contiguous"):
        vector_add(x[::2], y[::2])

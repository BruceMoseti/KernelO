"""verify() must accept correct low-precision results and reject realistic kernel bugs."""

import pytest
import torch

from kernelforge.testing import verify

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _matmul_inputs(
    m: int, n: int, k: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=generator).to(dtype)
    b = torch.randn(k, n, generator=generator).to(dtype)
    return a, b


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.double() @ b.double()


def _fp32_accumulated(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """What a correct kernel produces: fp32 accumulation, then rounding to the output dtype."""
    return (a.float() @ b.float()).to(a.dtype)


def _round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    bits = x.view(torch.int32)
    return ((bits + 0x1000) & ~0x1FFF).view(torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [16, 127, 1025, 4096])
def test_correct_matmul_passes_with_headroom(dtype: torch.dtype, k: int) -> None:
    a, b = _matmul_inputs(128, 96, k, dtype)
    result = verify(_reference(a, b), _fp32_accumulated(a, b), dtype)
    assert result.passed, result.describe()
    # Rounding to fp16/bf16 uses at most half of the one-ulp budget; fp32 uses far less.
    assert result.max_error_ratio < 0.6


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [127, 1025, 4096])
def test_dropped_k_term_fails(dtype: torch.dtype, k: int) -> None:
    a, b = _matmul_inputs(128, 96, k, dtype)
    buggy = (a[:, :-1].float() @ b[:-1].float()).to(dtype)
    assert not verify(_reference(a, b), buggy, dtype).passed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [127, 1025, 4096])
def test_low_precision_accumulation_fails(dtype: torch.dtype, k: int) -> None:
    a, b = _matmul_inputs(128, 96, k, dtype)
    accumulator = torch.zeros(128, 96, dtype=dtype)
    for k0 in range(0, k, 16):
        partial = a[:, k0 : k0 + 16].float() @ b[k0 : k0 + 16].float()
        accumulator = (accumulator.float() + partial).to(dtype)
    assert not verify(_reference(a, b), accumulator, dtype).passed


@pytest.mark.parametrize("k", [127, 1025, 4096])
def test_tf32_inputs_fail_the_ieee_fp32_check(k: int) -> None:
    a, b = _matmul_inputs(128, 96, k, torch.float32)
    tf32_result = _round_to_tf32(a) @ _round_to_tf32(b)
    assert not verify(_reference(a, b), tf32_result, torch.float32).passed


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_single_wrong_element_fails(dtype: torch.dtype) -> None:
    a, b = _matmul_inputs(129, 97, 255, dtype)
    reference = _reference(a, b)
    output = _fp32_accumulated(a, b)
    output.view(-1)[reference.abs().argmax()] = 0
    result = verify(reference, output, dtype)
    assert not result.passed
    assert result.mismatches == 1


def test_nan_output_fails() -> None:
    a, b = _matmul_inputs(32, 32, 64, torch.float16)
    output = _fp32_accumulated(a, b)
    output[0, 0] = float("nan")
    result = verify(_reference(a, b), output, torch.float16)
    assert not result.passed
    assert result.mismatches == 1
    assert result.max_error_ratio == float("inf")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [100, 1000, 4096, 32768])
def test_correct_softmax_passes(dtype: torch.dtype, n: int) -> None:
    # With n=32768, fp16 outputs are subnormal and only representable to the subnormal spacing;
    # without the spacing term in the tolerance, this correct result would be rejected.
    x = torch.randn(8, n, generator=torch.Generator().manual_seed(0)).to(dtype)
    output = torch.softmax(x.float(), dim=-1).to(dtype)
    result = verify(torch.softmax(x.double(), dim=-1), output, dtype)
    assert result.passed, result.describe()
    assert result.max_error_ratio < 0.6


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_with_zero_filled_padding_fails(dtype: torch.dtype) -> None:
    # A masking bug: loading out-of-range lanes as 0 instead of -inf adds exp(0 - max) per
    # padding lane to the denominator.
    x = torch.randn(8, 100, generator=torch.Generator().manual_seed(0)).to(dtype)
    padded = torch.cat([x.float(), torch.zeros(8, 28)], dim=-1)
    buggy = torch.softmax(padded, dim=-1)[:, :100].to(dtype)
    assert not verify(torch.softmax(x.double(), dim=-1), buggy, dtype).passed


@pytest.mark.parametrize("dtype", DTYPES)
def test_elementwise_add_passes_and_an_unwritten_tail_fails(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1000, generator=generator).to(dtype)
    y = torch.randn(1000, generator=generator).to(dtype)
    reference = x.double() + y.double()
    output = x + y
    assert verify(reference, output, dtype).passed
    output[-1] += 1
    assert not verify(reference, output, dtype).passed


@pytest.mark.parametrize("dtype", DTYPES)
def test_check_is_scale_invariant(dtype: torch.dtype) -> None:
    a, b = _matmul_inputs(64, 64, 512, dtype)
    scale = 2.0**6
    base = verify(_reference(a, b), _fp32_accumulated(a, b), dtype)
    scaled = verify(_reference(a * scale, b), _fp32_accumulated(a * scale, b), dtype)
    assert scaled.max_error_ratio == pytest.approx(base.max_error_ratio, rel=1e-6)


def test_rejects_misuse() -> None:
    reference = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="float64"):
        verify(reference.float(), reference.float(), torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        verify(reference, reference.half(), torch.float32)
    with pytest.raises(ValueError, match="shape"):
        verify(reference, torch.zeros(5), torch.float32)
    with pytest.raises(ValueError, match="non-finite"):
        verify(reference.clone().fill_(float("inf")), torch.zeros(4), torch.float32)

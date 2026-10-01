"""scripts/gpu_validate.sh must refuse to run where its results would not be GPU results."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_validate.sh"


def _run(**env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=300,
    )


def test_refuses_to_validate_interpreted_kernels():
    result = _run(TRITON_INTERPRET="1")
    assert result.returncode == 1
    assert "unset TRITON_INTERPRET" in result.stderr


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the refusal without a GPU")
def test_refuses_to_run_without_a_gpu():
    result = _run(TRITON_INTERPRET="0")
    assert result.returncode == 1
    assert "needs an NVIDIA GPU" in result.stderr

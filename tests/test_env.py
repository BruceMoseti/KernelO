"""Environment snapshot tests: where the recorded provenance comes from."""

from __future__ import annotations

import subprocess

import pytest
import torch

from kernelforge.runtime import env


@pytest.fixture
def driver_version():
    env._driver_version.cache_clear()
    yield env._driver_version
    env._driver_version.cache_clear()


def test_driver_version_is_the_nvidia_driver_release(monkeypatch, driver_version):
    """Not the CUDA driver API version (12040 -> "12.4"), which many releases share."""
    monkeypatch.setattr(torch._C, "_cuda_getDriverVersion", lambda: 12040, raising=False)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "550.54.15\n550.54.15\n"),
    )
    assert driver_version() == "550.54.15"


def test_driver_version_is_unknown_without_nvidia_smi(monkeypatch, driver_version):
    monkeypatch.setattr(torch._C, "_cuda_getDriverVersion", lambda: 12040, raising=False)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    assert driver_version() is None


@pytest.mark.parametrize("fp16, bf16", [(True, False), (False, True)])
def test_reduced_precision_reduction_flags_are_recorded(monkeypatch, fp16, bf16):
    """They change what cuBLAS computes for the fp16 and bf16 PyTorch baselines."""
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(matmul, "allow_fp16_reduced_precision_reduction", fp16)
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", bf16)
    snapshot = env.capture_environment()
    assert snapshot.torch_matmul_fp16_reduced_precision is fp16
    assert snapshot.torch_matmul_bf16_reduced_precision is bf16
    assert f"fp16 {'allowed' if fp16 else 'off'}, bf16 {'allowed' if bf16 else 'off'}" in (
        snapshot.render()
    )

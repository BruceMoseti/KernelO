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

import dataclasses

import pytest
import torch
import triton

from kernelforge.runtime.environment import collect_environment


def test_software_versions_are_recorded() -> None:
    env = collect_environment()
    assert env.torch == torch.__version__
    assert env.triton == triton.__version__
    assert env.cuda == torch.version.cuda
    assert env.python and env.platform and env.cpu


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only path")
def test_gpu_fields_are_none_without_cuda() -> None:
    env = collect_environment()
    gpu_fields = ("gpu", "compute_capability", "sm_count", "gpu_memory_bytes", "l2_cache_bytes")
    assert all(getattr(env, name) is None for name in gpu_fields)
    assert env.driver is None


@pytest.mark.gpu
def test_gpu_fields_are_populated() -> None:
    env = collect_environment()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert env.gpu == props.name
    assert env.compute_capability == f"{props.major}.{props.minor}"
    assert env.sm_count == props.multi_processor_count
    assert env.l2_cache_bytes == props.L2_cache_size
    assert env.driver, "nvidia-smi should report the driver version on a GPU machine"


def test_environment_is_serializable() -> None:
    record = dataclasses.asdict(collect_environment())
    assert set(record) >= {"gpu", "compute_capability", "driver", "cuda", "torch", "triton"}

from __future__ import annotations

from kernelforge.benchmark import workloads


def sweep_shapes(operation: str) -> set[tuple[int, int]]:
    return {
        (problem.dims_dict["rows"], problem.dims_dict["cols"])
        for problem in workloads.problems(operation, "sweep")
    }


def test_softmax_sweep_covers_its_benchmark_grid():
    grid = {
        (rows, cols)
        for rows in (128, 512, 2048, 8192)
        for cols in (128, 256, 512, 1024, 2048, 4096)
    }
    assert grid <= sweep_shapes("softmax")


def test_rmsnorm_sweep_covers_every_hidden_size():
    hidden_sizes = {cols for _, cols in sweep_shapes("rmsnorm")}
    assert {512, 768, 1024, 2048, 4096, 8192} <= hidden_sizes

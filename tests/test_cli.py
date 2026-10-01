"""CLI tests.

Argument parsing, shape resolution and the commands that work without a GPU
are covered here. The commands that launch kernels are thin wrappers over
library calls that are tested directly, and are marked ``gpu``.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.cli.main import OPERATIONS, build_parser, build_problem, main


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_every_operation_is_offered_by_the_cli():
    from kernelforge.kernels import operator_names

    assert set(OPERATIONS) == set(operator_names())


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "kernelforge" in capsys.readouterr().out


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        parse([])


def test_unknown_operation_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        parse(["tune", "flash_attention"])


def test_gemm_shape_flags_map_to_named_dimensions():
    args = parse(["tune", "matmul", "-m", "512", "-n", "1024", "-k", "256"])
    problem = build_problem("matmul", args)
    assert problem.dims_dict == {"M": 512, "N": 1024, "K": 256}
    assert problem.dtype is torch.float16


def test_row_shape_flags_map_to_named_dimensions():
    args = parse(["tune", "rmsnorm", "--rows", "128", "--cols", "768", "--dtype", "bf16"])
    problem = build_problem("rmsnorm", args)
    assert problem.dims_dict == {"rows": 128, "cols": 768}
    assert problem.dtype is torch.bfloat16


def test_vector_add_uses_its_own_flag():
    """``--n`` means N for a GEMM, so the elementwise count needs a distinct name."""
    args = parse(["tune", "vector_add", "--elements", "1000003"])
    assert build_problem("vector_add", args).dims_dict == {"n": 1000003}


@pytest.mark.parametrize("operation", OPERATIONS)
def test_defaults_produce_a_valid_problem(operation):
    """``kernelforge tune <op>`` works without shape flags."""
    problem = build_problem(operation, parse(["tune", operation]))
    assert problem.operation == operation
    assert all(value > 0 for value in problem.dims_dict.values())


def test_unsupported_dtype_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        parse(["tune", "matmul", "--dtype", "fp8"])


def test_env_command_runs_without_a_gpu(capsys):
    assert main(["env"]) == 0
    out = capsys.readouterr().out
    assert "PyTorch" in out
    if not torch.cuda.is_available():
        assert "No CUDA device" in out


def test_explain_reports_the_search_space_without_a_gpu(capsys):
    assert main(["tune", "matmul", "-m", "2048", "-n", "4096", "-k", "4096", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "grid points" in out
    assert "selected (budget)" in out


def test_tune_fails_clearly_without_a_gpu(capsys):
    if torch.cuda.is_available():
        pytest.skip("this is the no-GPU error path")
    assert main(["tune", "matmul", "--no-db", "--no-cache"]) == 1
    assert "CUDA" in capsys.readouterr().err


def test_report_without_a_database_is_an_error(capsys, tmp_path):
    assert main(["report", "--db", str(tmp_path / "missing.db")]) == 2
    assert "no results database" in capsys.readouterr().err


def test_cache_list_on_an_empty_cache(capsys, tmp_path):
    assert main(["cache", "list", "--cache", str(tmp_path / "configs.json")]) == 0
    assert "0 entry(ies)" in capsys.readouterr().out


def test_cache_clear_reports_what_it_removed(capsys, tmp_path):
    from kernelforge.tuning.cache import ConfigCache
    from kernelforge.tuning.config import KernelConfig, Problem

    path = tmp_path / "configs.json"
    cache = ConfigCache(path)
    cache.put(
        Problem.create("matmul", "fp16", M=64, N=64, K=64),
        "NVIDIA_Test_sm80",
        KernelConfig("matmul", BLOCK_M=64),
    )
    assert main(["cache", "list", "--cache", str(path)]) == 0
    assert "1 entry(ies)" in capsys.readouterr().out

    assert main(["cache", "clear", "--cache", str(path)]) == 0
    assert "removed 1 cached configuration" in capsys.readouterr().out
    assert not path.exists()


def test_nsight_child_argv_preserves_the_requested_shape():
    """The ncu target must profile the shape that was asked for.

    Rebuilding the child command from the parsed arguments rather than slicing
    ``sys.argv`` is what makes this hold when a shape flag follows
    ``--backend``.
    """
    from kernelforge.cli.main import nsight_child_argv

    args = parse(
        ["profile", "matmul", "--backend", "nsight", "-m", "512", "-n", "1024", "-k", "256"]
    )
    child = nsight_child_argv(args)
    assert child[:2] == ["profile", "matmul"]
    for flag, value in (("-m", "512"), ("-n", "1024"), ("-k", "256")):
        assert child[child.index(flag) + 1] == value
    # The child runs exactly one launch and records nothing.
    assert child[-2:] == ["--backend", "launch-once"]
    assert "--no-db" in child
    assert child.count("--backend") == 1


def test_nsight_child_argv_passes_the_cache_through():
    """The child has to select the same configuration as the parent."""
    from kernelforge.cli.main import nsight_child_argv

    args = parse(["profile", "rmsnorm", "--rows", "64", "--cols", "128", "--cache", "/tmp/kf.json"])
    child = nsight_child_argv(args)
    assert child[child.index("--cache") + 1] == "/tmp/kf.json"

    args = parse(["profile", "rmsnorm", "--no-cache"])
    assert "--no-cache" in nsight_child_argv(args)
    assert "--cache" not in nsight_child_argv(args)


def test_nsight_self_command_targets_the_module_entry_point():
    from kernelforge.profiling.nsight import self_command

    command = self_command(["profile", "matmul"])
    assert command[1:3] == ["-m", "kernelforge.cli.main"]


def test_nsight_sections_are_validated():
    from kernelforge.profiling.nsight import SECTIONS, build_command

    command = build_command(["python", "-c", "pass"], sections=("occupancy", "memory"))
    assert "--section" in command
    assert SECTIONS["occupancy"] in command
    with pytest.raises(ValueError, match="unknown sections"):
        build_command(["python"], sections=("not_a_section",))


def test_profile_reports_a_missing_ncu(capsys, monkeypatch):
    from kernelforge.profiling import nsight

    monkeypatch.setattr(nsight, "ncu_available", lambda: False)
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to reach the ncu check")
    assert main(["profile", "matmul", "--backend", "nsight"]) == 2
    assert "ncu not found" in capsys.readouterr().err


@pytest.mark.gpu
def test_compare_runs_end_to_end(capsys):
    assert (
        main(
            [
                "compare",
                "rmsnorm",
                "--rows",
                "512",
                "--cols",
                "1024",
                "--iterations",
                "20",
                "--warmup",
                "5",
                "--no-db",
                "--no-cache",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "kernelforge" in out
    assert "torch_eager" in out


@pytest.mark.gpu
def test_tune_runs_end_to_end_and_writes_both_stores(capsys, tmp_path):
    db_path = tmp_path / "results.db"
    cache_path = tmp_path / "configs.json"
    code = main(
        [
            "tune",
            "rmsnorm",
            "--rows",
            "512",
            "--cols",
            "1024",
            "--iterations",
            "20",
            "--warmup",
            "5",
            "--db",
            str(db_path),
            "--cache",
            str(cache_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "passed correctness" in out
    assert "Best configuration" in out
    assert db_path.exists() and cache_path.exists()

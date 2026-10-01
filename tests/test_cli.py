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


#: ``ncu --page raw --csv`` output as NVIDIA's Nsight Compute forum moderator
#: posted it (forums.developer.nvidia.com/t/220320), behind the ``==PROF==``
#: lines that the Nsight Compute CLI documentation shows on the same stream.
NCU_RAW_CSV = """\
==PROF== Connected to process 5268
==PROF== Profiling "vectorAdd_A" - 0: 0%....50%....100% - 46 passes
==PROF== Disconnected from process 5268
"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","launch__grid_size","sm__warps_active.avg.pct_of_peak_sustained_active"
"","","","","","","","","","%"
"0","12440","vectorAdd.exe","127.0.0.1","vectorAdd(const float *, const float *, float *, int)","2021-Nov-08 19:56:53","1","7","196","77.969567"
"""


def ncu_printing(monkeypatch, stdout: str):
    import subprocess

    from kernelforge.profiling import nsight

    monkeypatch.setattr(nsight, "ncu_available", lambda: True)
    monkeypatch.setattr(
        nsight.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout, ""),
    )
    return nsight.run(["./vectorAdd"])


def test_nsight_reads_the_raw_page_csv(monkeypatch):
    run = ncu_printing(monkeypatch, NCU_RAW_CSV)
    assert run.metrics == {
        "0: vectorAdd(const float *, const float *, float *, int)": {
            "launch__grid_size": "196",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "77.969567 %",
        }
    }


def test_nsight_keeps_each_launch_of_a_kernel(monkeypatch):
    second_launch = NCU_RAW_CSV.splitlines()[-1].replace('"0"', '"1"', 1)
    run = ncu_printing(monkeypatch, NCU_RAW_CSV + second_launch + "\n")
    assert list(run.metrics) == [
        "0: vectorAdd(const float *, const float *, float *, int)",
        "1: vectorAdd(const float *, const float *, float *, int)",
    ]


@pytest.mark.gpu
def test_nsight_reads_the_csv_a_real_ncu_prints():
    """The fixtures above follow NVIDIA's published example; this checks a real ncu."""
    import sys

    from kernelforge.profiling import nsight

    if not nsight.ncu_available():
        pytest.skip("Nsight Compute (ncu) is not on PATH")
    script = "import torch; torch.ones(8, device='cuda').add_(1); torch.cuda.synchronize()"
    run = nsight.run([sys.executable, "-c", script], sections=("launch",))
    assert run.ok, run.stderr
    assert run.metrics, run.stdout


def test_profile_reports_a_missing_ncu(capsys, monkeypatch):
    from kernelforge.profiling import nsight

    monkeypatch.setattr(nsight, "ncu_available", lambda: False)
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to reach the ncu check")
    assert main(["profile", "matmul", "--backend", "nsight"]) == 2
    assert "ncu not found" in capsys.readouterr().err


@pytest.mark.parametrize("impl", ["kernelforge", "torch_eager"])
def test_nsight_profiles_every_kernel_of_one_call(impl, monkeypatch):
    """ncu profiles only inside the child's profiler range, and all of it.

    The range opens after the inputs exist, so ``torch.randn`` is outside it,
    and it holds every kernel one call launches: the fused kernel alone, or
    the unfused sequence's GEMM, bias add and GELU together.
    """
    import subprocess

    from kernelforge.profiling import nsight

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(nsight, "ncu_available", lambda: True)
    monkeypatch.setattr(nsight.subprocess, "run", fake_run)
    argv = ["profile", "fused_linear", "--backend", "nsight", "--impl", impl, "--no-cache"]
    assert main(argv) == 0

    (command,) = commands
    assert command[command.index("--profile-from-start") + 1] == "off"
    assert "--launch-count" not in command
    assert command[command.index("--metrics") + 1] == "dram__bytes_read.sum,dram__bytes_write.sum"
    child = command[command.index("kernelforge.cli.main") + 1 :]
    assert child[child.index("--impl") + 1] == impl


@pytest.mark.parametrize("impl", ["kernelforge", "torch_eager"])
def test_profiled_child_runs_one_warmed_call_inside_the_range(impl, monkeypatch, tmp_path):
    """Inputs first, then one call to compile and initialise, then one call profiled."""
    import contextlib

    import kernelforge.kernels
    import kernelforge.runtime.env
    from kernelforge.tuning.config import KernelConfig

    events = []

    class Recorder:
        def make_inputs(self, problem, device, *, seed=0):
            events.append("inputs")
            return ()

        def default_config(self, problem):
            return KernelConfig("vector_add", BLOCK_SIZE=1024, num_warps=4)

        def run(self, config, *inputs):
            events.append("kernelforge")

        def baselines(self, problem, inputs):
            return {"torch_eager": lambda: events.append("torch_eager")}

    @contextlib.contextmanager
    def profiled_range():
        events.append("start")
        yield
        events.append("stop")

    monkeypatch.setattr(kernelforge.kernels, "get_operator", lambda name: Recorder())
    monkeypatch.setattr(kernelforge.runtime.env, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda.profiler, "profile", profiled_range)
    argv = ["profile", "vector_add", "--backend", "launch-once", "--impl", impl]
    assert main([*argv, "--cache", str(tmp_path / "configs.json")]) == 0
    assert events == ["inputs", impl, "start", impl, "stop"]


def test_nsight_totals_dram_traffic_over_every_launch(monkeypatch):
    header = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
        '"Context","Stream","dram__bytes_read.sum","dram__bytes_write.sum"'
    )
    units = '"","","","","","","","","byte","byte"'
    launches = [
        f'"{i}","4242","python3","127.0.0.1","{name}","2026-Oct-01 05:00:00","1","7",'
        f'"{read}","{written}"'
        for i, (name, read, written) in enumerate(
            [("gemm", 3 * 2**20, 2**20), ("add", 2**20, 2**20), ("gelu", 2**20, 2**20)]
        )
    ]
    run = ncu_printing(monkeypatch, "\n".join([header, units, *launches]) + "\n")
    assert "DRAM over 3 launch(es): 5.0 MiB read, 3.0 MiB written" in run.render()


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


@pytest.mark.gpu
def test_nsight_counters_come_from_the_kernelforge_kernel(monkeypatch):
    from kernelforge.profiling import nsight

    if not nsight.ncu_available():
        pytest.skip("Nsight Compute (ncu) is not on PATH")
    runs = []
    real_run = nsight.run

    def recording_run(*args, **kwargs):
        runs.append(real_run(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(nsight, "run", recording_run)
    shape = ["-m", "256", "-n", "256", "-k", "256"]
    assert main(["profile", "matmul", *shape, "--backend", "nsight", "--no-cache"]) == 0
    (run,) = runs
    assert "matmul_kernel" in run.stdout


@pytest.mark.gpu
def test_nsight_profiles_every_kernel_of_the_unfused_sequence(monkeypatch):
    from kernelforge.profiling import nsight

    if not nsight.ncu_available():
        pytest.skip("Nsight Compute (ncu) is not on PATH")
    runs = []
    real_run = nsight.run

    def recording_run(*args, **kwargs):
        runs.append(real_run(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(nsight, "run", recording_run)
    shape = ["-m", "256", "-n", "256", "-k", "256"]
    argv = ["profile", "fused_linear", *shape, "--backend", "nsight", "--impl", "torch_eager"]
    assert main([*argv, "--no-cache"]) == 0
    (run,) = runs
    # The GEMM, the bias add and the GELU, at least.
    assert len(run.metrics) >= 3, run.stdout
    assert "DRAM over" in run.render()

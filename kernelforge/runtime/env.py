"""Hardware and software provenance for every measurement.

A latency number without hardware context is not a result, it is an anecdote.
Everything KernelForge records -- tuning runs, benchmark rows, cache entries --
is stamped with the snapshot produced here.
"""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch


@dataclass(frozen=True)
class DeviceCaps:
    """The subset of device properties the search-space filters reason about.

    Kept as a plain dataclass rather than reading ``torch.cuda`` directly so
    that feasibility filters can be unit tested on a CPU-only machine against
    the properties of a GPU that is not attached.
    """

    name: str
    compute_capability: str
    total_memory_bytes: int
    sm_count: int
    max_shared_memory_per_block: int
    shared_memory_per_sm: int
    registers_per_sm: int
    max_threads_per_sm: int
    warp_size: int
    l2_cache_bytes: int
    #: False when these values were not read from an attached device. Filters
    #: reason over whichever properties they are given, so anything reporting
    #: their output has to say which device it was reasoning about -- a
    #: candidate count derived from assumed properties is not a measurement.
    measured: bool = True

    @property
    def arch(self) -> str:
        return "sm" + self.compute_capability.replace(".", "")

    @property
    def key(self) -> str:
        """Stable identifier used in cache keys.

        Deliberately includes the full board name and not just the compute
        capability: an RTX 4090 and an RTX 4080 are both ``sm89`` but differ in
        SM count, L2 size and memory bandwidth, so a configuration tuned on one
        is not necessarily best on the other.
        """
        safe = "".join(c if c.isalnum() else "_" for c in self.name).strip("_")
        while "__" in safe:
            safe = safe.replace("__", "_")
        return f"{safe}_{self.arch}"


# Values for an A100-80GB, used only when a caller asks for capabilities while
# no CUDA device is attached (documentation builds, filter unit tests, CI).
# `measured=False` so that output derived from them can label itself.
_FALLBACK_CAPS = DeviceCaps(
    name="A100-SXM4-80GB (assumed)",
    compute_capability="8.0",
    total_memory_bytes=80 * 1024**3,
    sm_count=108,
    max_shared_memory_per_block=166912,
    shared_memory_per_sm=167936,
    registers_per_sm=65536,
    max_threads_per_sm=2048,
    warp_size=32,
    l2_cache_bytes=40 * 1024 * 1024,
    measured=False,
)


def device_caps(device: torch.device | str | int | None = None) -> DeviceCaps:
    """Read capabilities off a CUDA device, or return the documented fallback."""
    if not torch.cuda.is_available():
        return _FALLBACK_CAPS
    index = 0 if device is None else torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    return DeviceCaps(
        name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        total_memory_bytes=int(props.total_memory),
        sm_count=int(props.multi_processor_count),
        # `shared_memory_per_block_optin` is the limit a kernel can opt in to
        # via dynamic shared memory, which is what Triton uses; the plain
        # `shared_memory_per_block` is the 48 KiB static default.
        max_shared_memory_per_block=int(
            getattr(
                props,
                "shared_memory_per_block_optin",
                getattr(props, "shared_memory_per_block", 49152),
            )
        ),
        shared_memory_per_sm=int(getattr(props, "shared_memory_per_multiprocessor", 65536)),
        registers_per_sm=int(getattr(props, "regs_per_multiprocessor", 65536)),
        max_threads_per_sm=int(getattr(props, "max_threads_per_multi_processor", 2048)),
        warp_size=int(getattr(props, "warp_size", 32)),
        l2_cache_bytes=int(getattr(props, "L2_cache_size", 0)) or 4 * 1024 * 1024,
    )


def device_key(device: torch.device | str | int | None = None) -> str:
    """The cache key for the attached device, or ``"cpu"`` when there is none.

    The single derivation used by both the recorded environment and the
    dispatch-time cache lookup. Deriving it twice let them disagree on a
    CPU-only host, where `device_caps` returns documented A100 fallback values
    whose key would have claimed hardware that is not present.
    """
    if not torch.cuda.is_available():
        return "cpu"
    return device_caps(device).key


def _triton_version() -> str | None:
    try:
        import triton
    except Exception:
        return None
    return getattr(triton, "__version__", None)


def _driver_version() -> str | None:
    try:
        raw = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
    except Exception:
        raw = None
    if isinstance(raw, int) and raw > 0:
        return f"{raw // 1000}.{(raw % 1000) // 10}"
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip().splitlines()[0].strip() or None


@dataclass(frozen=True)
class Environment:
    timestamp: str
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    triton_version: str | None
    cuda_version: str | None
    driver_version: str | None
    gpu_name: str | None
    gpu_arch: str | None
    gpu_memory_bytes: int | None
    sm_count: int | None
    device_key: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        rows = [
            ("GPU", self.gpu_name or "none (CPU only)"),
            ("Compute capability", self.gpu_arch or "-"),
            ("SM count", str(self.sm_count) if self.sm_count else "-"),
            (
                "VRAM",
                f"{self.gpu_memory_bytes / 1024**3:.1f} GiB" if self.gpu_memory_bytes else "-",
            ),
            ("CUDA", self.cuda_version or "-"),
            ("Driver", self.driver_version or "-"),
            ("PyTorch", self.torch_version),
            ("Triton", self.triton_version or "not installed"),
            ("Python", self.python_version),
            ("Platform", self.platform),
            ("Host", self.hostname),
            ("Timestamp", self.timestamp),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


def capture_environment(device: torch.device | str | int | None = None) -> Environment:
    has_cuda = torch.cuda.is_available()
    caps = device_caps(device) if has_cuda else None
    return Environment(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        triton_version=_triton_version(),
        cuda_version=torch.version.cuda if has_cuda else None,
        driver_version=_driver_version() if has_cuda else None,
        gpu_name=caps.name if caps else None,
        gpu_arch=caps.compute_capability if caps else None,
        gpu_memory_bytes=caps.total_memory_bytes if caps else None,
        sm_count=caps.sm_count if caps else None,
        device_key=device_key(device),
    )


def require_cuda() -> torch.device:
    """Fail loudly and early rather than silently measuring CPU latency."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "KernelForge kernels require a CUDA device. torch.cuda.is_available() returned False."
        )
    return torch.device("cuda")

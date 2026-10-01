# Reproducible environment for KernelForge.
#
# Built on the official PyTorch CUDA image, which already carries a matched
# torch / Triton / nvcc set. Pinning that combination is the point: a tuning
# result is only comparable against another one produced by the same compiler,
# and "install the latest PyTorch" does not pin anything.
#
#   docker build -t kernelforge .
#   docker run --rm --gpus all kernelforge kernelforge env
#   docker run --rm --gpus all kernelforge make test-all
#
# The CPU-side suite, including the kernel compilation checks, runs without
# --gpus all:
#
#   docker run --rm kernelforge make test
#
# `--gpus all` needs the NVIDIA Container Toolkit on the host.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# clang and a libstdc++ that clang can target NVPTX with are needed by the
# CUDA device-code compilation checks, which run without a GPU. libstdc++ 14
# declares numeric_limits<__float128>, which NVPTX has no such type for, so
# version 13 is installed explicitly; the test harness probes for a working
# one and skips if it finds none.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang libstdc++-13-dev git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kernelforge

# Dependency metadata first, so editing a kernel does not invalidate the
# dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY kernelforge/__init__.py kernelforge/
RUN pip install --no-cache-dir -e '.[dev,report]'

COPY . .

# Keep generated measurements inside the container unless a volume is mounted,
# so a stray run cannot mix results from two machines into one database.
ENV KERNELFORGE_DB=/opt/kernelforge/results/kernelforge.db \
    KERNELFORGE_CACHE=/opt/kernelforge/results/configs.json

CMD ["make", "test"]
